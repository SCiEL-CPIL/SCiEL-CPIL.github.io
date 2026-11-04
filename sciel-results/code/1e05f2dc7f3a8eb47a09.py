import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap.
    """
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _compute_side(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    Uses support function: side = max over 6 directions of (max projection of inner hex vertices)
    For a regular hexagon with side s centered at origin, the support in direction theta is s*cos(theta - vertex_angle).
    Actually: for outer hex with side S and rotation phi, a point (px,py) is inside iff
    for all 6 face normals n_k: n_k . (p - center) <= S * cos(30deg) = S * sqrt(3)/2
    Wait, let me think. The outer hex has vertices at S*cos(60k+phi), S*sin(60k+phi).
    The face between vertex k and k+1 has normal at angle 60(k+0.5)+phi = 30+60k+phi.
    The distance from center to each face is S*cos(30deg) = S*sqrt(3)/2.
    So point p is inside iff for all k: dot(p - center, n_k) <= S*sqrt(3)/2
    where n_k = (cos(30+60k+phi), sin(30+60k+phi)).
    So S >= max_k [ dot(p - center, n_k) / (sqrt(3)/2) ] for all inner hex vertices p.
    """
    n = len(inner_params)
    phi = np.deg2rad(outer_angle_deg)
    # Face normals of outer hex
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    # apothem factor
    apothem_factor = SQRT3 / 2.0
    
    max_proj = 0.0
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        # projections of all 6 vertices onto each of 6 normals
        projs = hex_v @ normals.T  # shape (6, 6)
        # for each normal, the max projection must be <= S * apothem_factor
        max_projs = np.max(projs, axis=0)  # shape (6,)
        required_sides = max_projs / apothem_factor
        max_proj = max(max_proj, np.max(required_sides))
    
    return max_proj

def _overlap_penalty(inner_params, penalty_weight=200.0, threshold=0.05):
    """
    Compute penalty for overlapping inner hexagons using vectorized distance checks.
    """
    n = len(inner_params)
    penalty = 0.0
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(inner_params[i, 0], inner_params[i, 1], inner_params[i, 2])
    
    for i in range(n):
        for j in range(i+1, n):
            # Quick bounding box check
            dx = inner_params[j, 0] - inner_params[i, 0]
            dy = inner_params[j, 1] - inner_params[i, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > 16.0:  # Max distance for hexagons to possibly overlap
                continue
            
            sep = _poly_separation(verts[i], verts[j])
            if sep < threshold:
                penalty += penalty_weight * (threshold - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """
    Objective function: outer side length + overlap penalty.
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _objective_smooth(params, outer_angle_deg, n, beta=50.0):
    """
    Smooth objective using soft-max for side computation.
    """
    inner_params = params.reshape(n, 3)
    phi = np.deg2rad(outer_angle_deg)
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = SQRT3 / 2.0
    
    all_projs = []
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        projs = hex_v @ normals.T  # shape (6, 6)
        max_projs = np.max(projs, axis=0)  # shape (6,)
        required_sides = max_projs / apothem_factor
        all_projs.append(required_sides)
    
    all_required = np.array(all_projs)  # shape (n, 6)
    # Soft-max over all entries
    all_flat = all_required.flatten()
    # Use log-sum-exp for numerical stability
    m = np.max(all_flat)
    soft_max = m + np.log(np.sum(np.exp(all_flat - m))) / beta
    
    overlap_pen = _overlap_penalty(inner_params)
    return soft_max + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    
    return True

def _seed_solution():
    """Return an optimized seed solution using hexagonal lattice with slight adjustments."""
    eps = 1e-9
    centers = []
    # Center
    centers.append((0.0, 0.0))
    # First ring: 6 hexagons at distance sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (SQRT3 + eps) * np.cos(angle)
        y = (SQRT3 + eps) * np.sin(angle)
        centers.append((x, y))
    # Second ring: 4 hexagons at distance 3, offset by 30 degrees
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # Compute tight bounding hexagon
    all_verts = []
    for x, y, a in inner:
        verts = _hex_vertices(x, y, a)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Find minimum enclosing regular hexagon
    best_side = float('inf')
    for outer_angle in [0, 30]:
        normals = np.column_stack([
            np.cos(np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle)),
            np.sin(np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle))
        ])
        projs = all_verts @ normals.T
        max_projs = np.max(projs, axis=0)
        side = np.max(max_projs) / (SQRT3 / 2.0)
        best_side = min(best_side, side)
    
    side = best_side + 1e-9
    return inner, (0.0, 0.0), side, 30.0

def _psp_swarm(inner_params, outer_angle_deg, n, iterations=300, swarm_size=30, time_limit=30.0):
    """
    Particle Swarm Optimization to minimize the objective.
    Uses smooth objective for faster evaluation.
    Returns the best particle found.
    """
    start_time = time.time()
    best_global = inner_params.copy()
    best_global_val = _objective(best_global.flatten(), outer_angle_deg, n)
    
    # Initialize swarm
    particles = np.zeros((swarm_size, n, 3))
    velocities = np.zeros((swarm_size, n, 3))
    personal_bests = np.zeros((swarm_size, n, 3))
    personal_best_vals = np.zeros(swarm_size)
    
    for i in range(swarm_size):
        if i == 0:
            particles[i] = inner_params
        else:
            # Use smaller perturbations for angles, larger for positions
            pert = np.random.randn(n, 3)
            pert[:, 2] *= 0.3  # smaller angle perturbations
            particles[i] = inner_params + pert * 0.4
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)
        
        val = _objective(particles[i].flatten(), outer_angle_deg, n)
        personal_bests[i] = particles[i]
        personal_best_vals[i] = val
        
        if val < best_global_val:
            best_global_val = val
            best_global = particles[i].copy()
    
    # PSO parameters
    w = 0.9  # inertia
    c1 = 1.5  # cognitive
    c2 = 1.5  # social
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        for i in range(swarm_size):
            r1 = np.random.rand(n, 3)
            r2 = np.random.rand(n, 3)
            
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_bests[i] - particles[i]) + c2 * r2 * (best_global - particles[i])
            
            # Clamp velocities - different for position and angle
            max_vel_pos = 0.3
            max_vel_ang = 0.1
            velocities[i, :, :2] = np.clip(velocities[i, :, :2], -max_vel_pos, max_vel_pos)
            velocities[i, :, 2] = np.clip(velocities[i, :, 2], -max_vel_ang, max_vel_ang)
            
            particles[i] = particles[i] + velocities[i]
            particles[i, :, 2] = np.mod(particles[i, :, 2], 360)
            
            val = _objective(particles[i].flatten(), outer_angle_deg, n)
            
            if val < personal_best_vals[i]:
                personal_best_vals[i] = val
                personal_bests[i] = particles[i].copy()
                
                if val < best_global_val:
                    best_global_val = val
                    best_global = particles[i].copy()
        
        # Decay inertia
        w = max(0.4, w * 0.999)
    
    return best_global, best_global_val

def _local_refine(inner_params, outer_angle_deg, n, time_limit=20.0):
    """
    Use Nelder-Mead for local refinement with smooth objective first,
    then polish with exact objective.
    """
    start_time = time.time()
    
    init_params = inner_params.flatten()
    
    # First pass: smooth objective for faster convergence
    try:
        res_smooth = minimize(_objective_smooth, init_params, args=(outer_angle_deg, n), 
                      method='Nelder-Mead', 
                      options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res_smooth.fun < _objective(init_params, outer_angle_deg, n):
            init_params = res_smooth.x
    except Exception:
        pass
    
    # Second pass: exact objective for final polish
    try:
        res = minimize(_objective, init_params, args=(outer_angle_deg, n), 
                      method='Nelder-Mead', 
                      options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if time.time() - start_time < time_limit:
            candidate = res.x.reshape(n, 3)
            if res.fun < _objective(inner_params.flatten(), outer_angle_deg, n):
                return candidate
    except Exception:
        pass
    
    return inner_params

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    safety_margin = 5.0
    
    n = 11
    outer_angle = 30.0
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = np.array(seed_inner)
    best_side = seed_side
    
    # Multi-start local optimization with smooth objective
    starts = [best_inner.copy()]
    # Add perturbed starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        pert = rng.randn(n, 3) * 0.3
        pert[:, 2] *= 0.5
        start = best_inner + pert
        start[:, 2] = np.mod(start[:, 2], 360)
        starts.append(start)
    
    for start_idx, start_params in enumerate(starts):
        remaining = DEADLINE - safety_margin - (time.time() - start_time)
        if remaining < 5:
            break
        
        init_params = start_params.flatten()
        try:
            res = minimize(_objective_smooth, init_params, args=(outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_side:
                candidate = res.x.reshape(n, 3)
                candidate_side = _compute_side(candidate, outer_angle)
                if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # PSO phase for global exploration
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 15:
        pso_time = min(remaining * 0.4, 35.0)
        try:
            pso_result, pso_val = _psp_swarm(best_inner, outer_angle, n, iterations=150, swarm_size=20, time_limit=pso_time)
            if pso_val < best_side:
                candidate = pso_result
                candidate_side = _compute_side(candidate, outer_angle)
                if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Local refinement phase - smooth then exact
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 8:
        try:
            refined = _local_refine(best_inner, outer_angle, n, time_limit=min(remaining * 0.7, 25.0))
            refined_side = _compute_side(refined, outer_angle)
            if refined_side < best_side and _is_valid(refined, refined_side, outer_angle, min_sep=1e-9):
                best_inner = refined
                best_side = refined_side
        except Exception:
            pass
    
    # Final high-precision polish with exact objective
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 5:
        try:
            final_params = best_inner.flatten()
            res_final = minimize(_objective, final_params, args=(outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if res_final.fun < best_side:
                candidate = res_final.x.reshape(n, 3)
                candidate_side = _compute_side(candidate, outer_angle)
                if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Ultra-precise final polish
    remaining = DEADLINE - safety_margin - (time.time() - start_time)
    if remaining > 3:
        try:
            final_params = best_inner.flatten()
            res_ultra = minimize(_objective, final_params, args=(outer_angle, n), 
                          method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_ultra.fun < best_side:
                candidate = res_ultra.x.reshape(n, 3)
                candidate_side = _compute_side(candidate, outer_angle)
                if _is_valid(candidate, candidate_side, outer_angle, min_sep=1e-9):
                    best_inner = candidate
                    best_side = candidate_side
        except Exception:
            pass
    
    # Final check
    final_side = _compute_side(best_inner, outer_angle)
    if not _is_valid(best_inner, final_side, outer_angle, min_sep=1e-9):
        # Fallback to seed
        best_inner = np.array(seed_inner)
        final_side = seed_side
    
    # Format output
    inner_rows = []
    for row in best_inner:
        inner_rows.append([float(row[0]), float(row[1]), float(row[2])])
        
    center = (0.0, 0.0)
    side = float(final_side)
    angle_deg = float(outer_angle)
    
    return inner_rows, center, side, angle_deg
