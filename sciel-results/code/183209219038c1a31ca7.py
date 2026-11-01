"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a list of (x, y) tuples."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

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
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
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
    
    # Collect all projections
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    # Log-sum-exp for smooth max
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * p_i)))
    # This approximates max(p_i) as beta -> infinity
    scaled = [beta * (p - max_proj) for p in projections]
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
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
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
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    # Outer hexagon vertices
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    # For each inner hexagon, check if all vertices are inside the outer hexagon
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            # Check if point is inside convex polygon using cross products
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                # Vector from p1 to p2
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                # Vector from p1 to point
                px = vx - p1[0]
                py = vy - p1[1]
                # Cross product
                cross = ex * py - ey * px
                # For a convex polygon with vertices in counter-clockwise order,
                # the point is inside if all cross products have the same sign
                # We assume the hexagon is oriented such that interior is to the left
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

def smooth_overlap_penalty(inner, weight=100.0, threshold=2.0):
    """
    Compute a smooth penalty for overlapping inner hexagons.
    Uses distance-based smooth penalty instead of SAT for better gradients.
    """
    n = len(inner)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            dx = cx1 - cx2
            dy = cy1 - cy2
            d = math.sqrt(dx*dx + dy*dy)
            if d < threshold:
                # Smooth penalty: quadratic in the "penetration" depth
                # For hexagons, minimum center distance for no overlap is ~2.0
                # Use a smooth approximation
                pen = (threshold - d) ** 2
                penalty += weight * pen
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice - use tighter, more symmetric arrangement
    # Center hexagon + 6 around it + 4 more in a compact cluster
    # Axial coordinates for hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian with tight packing
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [(x, y, 0.0) for (x, y) in centers]
    
    # Compute initial outer side length
    initial_S = compute_required_side(initial_inner, (0.0, 0.0), 0.0)
    
    best_result = None
    best_score = float('inf')
    
    # Verify initial configuration
    if check_validity(initial_inner, (0.0, 0.0), initial_S, 0.0):
        best_result = (initial_inner, (0.0, 0.0), initial_S, 0.0)
        best_score = initial_S
    else:
        # Fallback: expand slightly
        scale = 1.0 + 0.01
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        initial_inner = [(x, y, 0.0) for (x, y) in centers]
        initial_S = compute_required_side(initial_inner, (0.0, 0.0), 0.0)
        if check_validity(initial_inner, (0.0, 0.0), initial_S, 0.0):
            best_result = (initial_inner, (0.0, 0.0), initial_S, 0.0)
            best_score = initial_S
        else:
            # Final fallback
            initial_S += 0.5
            best_result = (initial_inner, (0.0, 0.0), initial_S, 0.0)
            best_score = initial_S
    
    # Phase 1: Optimize inner positions with fixed outer at origin
    # Use smooth objective for better gradients
    def objective_inner_smooth(params, beta=50.0):
        n = N
        inner = []
        for i in range(n):
            cx = params[3*i]
            cy = params[3*i+1]
            ang = np.degrees(params[3*i+2]) % 360
            inner.append((cx, cy, ang))
        
        S = compute_required_side_smooth(inner, (0.0, 0.0), 0.0, beta)
        
        # Use smooth distance-based penalty instead of SAT
        penalty = smooth_overlap_penalty(inner, weight=100.0, threshold=2.0)
        
        return S + penalty
    
    # Build initial conditions for inner-only optimization (3*N params)
    initial_conditions = []
    
    # IC1: Standard hexagonal lattice
    ic1 = np.zeros(3 * N)
    for i, (px, py) in enumerate(centers):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    initial_conditions.append(ic1)
    
    # IC2: Perturbed lattice
    np.random.seed(42)
    ic2 = ic1 + np.random.randn(3*N) * 0.15
    initial_conditions.append(ic2)
    
    # IC3: Tighter lattice
    tight_scale = 0.99
    tight_positions = []
    for q, r in axial_coords:
        x = 1.5 * tight_scale * q
        y = SQRT3 * tight_scale * (r + q / 2.0)
        tight_positions.append((x, y))
    ic3 = np.zeros(3 * N)
    for i, (px, py) in enumerate(tight_positions):
        ic3[3*i] = px
        ic3[3*i+1] = py
        ic3[3*i+2] = 0.0
    initial_conditions.append(ic3)
    
    # IC4: Rotated inner hexagons by 30 deg
    ic4 = ic1.copy()
    for i in range(N):
        ic4[3*i+2] = np.radians(30)
    initial_conditions.append(ic4)
    
    # IC5: Perturbed rotated
    ic5 = ic4 + np.random.randn(3*N) * 0.1
    initial_conditions.append(ic5)
    
    # IC6: Different perturbation seed
    np.random.seed(123)
    ic6 = ic1 + np.random.randn(3*N) * 0.2
    initial_conditions.append(ic6)
    
    # IC7: Compact cluster - shift ring-2 hexagons inward
    ic7 = np.zeros(3 * N)
    compact_positions = []
    for i, (q, r) in enumerate(axial_coords):
        if i == 0:
            x, y = 0.0, 0.0
        else:
            # Pull outer hexagons slightly inward
            x = 1.5 * 0.98 * q
            y = SQRT3 * 0.98 * (r + q / 2.0)
        compact_positions.append((x, y))
    for i, (px, py) in enumerate(compact_positions):
        ic7[3*i] = px
        ic7[3*i+1] = py
        ic7[3*i+2] = 0.0
    initial_conditions.append(ic7)
    
    for ic in initial_conditions:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(objective_inner_smooth, ic, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                
                # Compute optimal outer hexagon for this inner configuration
                S = compute_required_side(inner, (0.0, 0.0), 0.0)
                
                # Verify validity
                if check_validity(inner, (0.0, 0.0), S, 0.0):
                    best_score = S
                    best_result = (inner, (0.0, 0.0), S, 0.0)
        except Exception:
            pass
    
    # Phase 2: Joint optimization with outer parameters for refinement
    if time.time() < deadline - 45:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            def objective_full(params, beta=50.0):
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
                
                S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta)
                
                # Use smooth distance-based penalty
                penalty = smooth_overlap_penalty(inner, weight=100.0, threshold=2.0)
                
                return S + penalty
            
            res = minimize(objective_full, params_best, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead refinement on best result
    if time.time() < deadline - 25:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            def objective_full2(params):
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
                
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                # Use smooth distance-based penalty
                penalty = smooth_overlap_penalty(inner, weight=100.0, threshold=2.0)
                
                return S + penalty
            
            res = minimize(objective_full2, params_best, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Phase 4: High-precision polish pass with tighter tolerances
    if time.time() < deadline - 10 and best_result is not None:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            def objective_polish(params):
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
                
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                # Use smooth distance-based penalty
                penalty = smooth_overlap_penalty(inner, weight=200.0, threshold=2.0)
                
                return S + penalty
            
            res = minimize(objective_polish, params_best, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Final time check
    if time.time() > deadline:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
