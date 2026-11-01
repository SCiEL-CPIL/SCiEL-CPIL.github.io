"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    This provides better gradient information for optimization.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # For each inner hexagon and each normal, find the max projection
    # Then do log-sum-exp over all these maxes
    max_projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for nx, ny in normals:
            max_proj_this = -1e10
            for v in verts:
                dx = v[0] - ox
                dy = v[1] - oy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                proj = lx * nx + ly * ny
                if proj > max_proj_this:
                    max_proj_this = proj
            max_projections.append(max_proj_this)
    
    if not max_projections:
        return 0.0
    
    max_proj = max(max_projections)
    if max_proj < -1000:
        return 0.0
    
    # Use log-sum-exp with numerical stability
    scaled = [beta * (p - max_proj) for p in max_projections]
    # Clip to avoid overflow
    scaled = [max(-50, min(50, s)) for s in scaled]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts_list.append(get_hex_vertices(ix, iy, 1.0, math.radians(iangle)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def objective(params, use_smooth=True, beta=100.0):
    """Objective function: outer side length + penalty for overlaps.
    Uses a fast distance-based penalty instead of SAT for speed during optimization.
    """
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    if use_smooth:
        S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta)
    else:
        S = compute_required_side(inner, (ox, oy), outer_angle)
    
    # Fast distance-based penalty: for unit hexagons, minimum center distance is SQRT3
    # Use a soft penalty that's cheap to compute
    penalty = 0.0
    min_dist = SQRT3  # minimum center-to-center distance for non-overlapping unit hexagons
    
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, _ = inner[i]
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < min_dist:
                # Quadratic penalty for violation
                penalty += (min_dist - d) ** 2 * 50
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Axial coordinates for 11 hexagons in a compact cluster
    # Center + 6 in ring 1 + 4 in ring 2
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    # Generate diverse initial conditions
    initial_conditions = []
    
    # Arrangement 1: Standard hexagonal lattice, scale 1.0
    scale1 = 1.0 + 1e-6
    centers1 = [axial_to_cartesian(q, r, scale1) for q, r in axial_coords]
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers1):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    ic1[3*N] = 0.0
    ic1[3*N+1] = 0.0
    ic1[3*N+2] = 0.0
    initial_conditions.append(ic1)
    
    # Arrangement 2: Rotated outer hexagon by 30 degrees
    ic2 = ic1.copy()
    ic2[3*N+2] = np.radians(30)
    initial_conditions.append(ic2)
    
    # Arrangement 3: Slightly tighter scale
    scale3 = 0.98
    centers3 = [axial_to_cartesian(q, r, scale3) for q, r in axial_coords]
    ic3 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers3):
        ic3[3*i] = px
        ic3[3*i+1] = py
        ic3[3*i+2] = 0.0
    ic3[3*N] = 0.0
    ic3[3*N+1] = 0.0
    ic3[3*N+2] = 0.0
    initial_conditions.append(ic3)
    
    # Arrangement 4: Perturbed standard lattice
    np.random.seed(42)
    ic4 = ic1 + np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic4)
    
    # Arrangement 5: Different outer angle
    ic5 = ic1.copy()
    ic5[3*N+2] = np.radians(15)
    initial_conditions.append(ic5)
    
    # Arrangement 6: 30-degree rotated inner hexagons with 30-degree outer
    ic6 = ic1.copy()
    for i in range(N):
        ic6[3*i+2] = np.radians(30)
    ic6[3*N+2] = np.radians(30)
    initial_conditions.append(ic6)
    
    # Arrangement 7: Slightly compressed vertically
    ic7 = ic1.copy()
    for i in range(N):
        ic7[3*i+1] *= 0.95
    initial_conditions.append(ic7)
    
    # Arrangement 8: Different subset of ring 2 positions
    axial_coords_alt = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (2, -2), (-2, 1)
    ]
    if len(axial_coords_alt) > N:
        axial_coords_alt = axial_coords_alt[:N]
    scale8 = 1.0 + 1e-6
    centers8 = [axial_to_cartesian(q, r, scale8) for q, r in axial_coords_alt]
    ic8 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers8):
        ic8[3*i] = px
        ic8[3*i+1] = py
        ic8[3*i+2] = 0.0
    ic8[3*N] = 0.0
    ic8[3*N+1] = 0.0
    ic8[3*N+2] = 0.0
    initial_conditions.append(ic8)
    
    # Arrangement 9: More perturbation
    np.random.seed(123)
    ic9 = ic1 + np.random.randn(3*N+3) * 0.1
    initial_conditions.append(ic9)
    
    best_result = None
    best_score = float('inf')
    
    # Optimize each initial condition
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 25:
            break
        
        try:
            # First pass: smooth objective with moderate beta
            res = minimize(lambda p: objective(p, use_smooth=True, beta=50.0), ic, 
                          method='L-BFGS-B', 
                          options={'maxiter': 120, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Second pass: refine with higher beta for sharper max approximation
            res_mid = minimize(lambda p: objective(p, use_smooth=True, beta=200.0), res.x, 
                              method='L-BFGS-B', 
                              options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
            
            # Third pass: refine with exact objective
            res2 = minimize(lambda p: objective(p, use_smooth=False), res_mid.x, 
                           method='L-BFGS-B', 
                           options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
            
            # Use the best of the three
            best_res = min([res, res_mid, res2], key=lambda r: r.fun)
            
            if best_res.fun < best_score or best_result is None:
                params = best_res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        centers_fb = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in centers_fb]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Refine if time permits - multi-stage polish
    if time.time() < deadline - 35:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            # Stage 1: L-BFGS-B with smooth objective
            res1 = minimize(lambda p: objective(p, use_smooth=True, beta=300.0), params_best, 
                          method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            
            # Stage 2: L-BFGS-B with exact objective
            res2 = minimize(lambda p: objective(p, use_smooth=False), res1.x, 
                          method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            
            # Stage 3: Nelder-Mead polish
            res3 = minimize(lambda p: objective(p, use_smooth=False), res2.x,
                          method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            
            # Stage 4: Final L-BFGS-B
            res4 = minimize(lambda p: objective(p, use_smooth=False), res3.x,
                          method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            
            # Pick the best
            best_res = min([res1, res2, res3, res4], key=lambda r: r.fun)
            
            if best_res.fun < best_score:
                params = best_res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Final Nelder-Mead polish on the best result
    if time.time() < deadline - 15 and best_result is not None:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res_nm = minimize(lambda p: objective(p, use_smooth=False), params_best,
                            method='Nelder-Mead',
                            options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-13})
            
            if res_nm.fun < best_score:
                params = res_nm.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
