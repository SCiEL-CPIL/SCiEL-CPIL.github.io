"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(angle_rad=0.0, center=None):
    """Return 6 vertices of a unit regular hexagon rotated by angle_rad, optionally translated by center."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = math.cos(theta)
        y = math.sin(theta)
        if center is not None:
            x += center[0]
            y += center[1]
        verts.append([x, y])
    return np.array(verts)

def _get_aabb(poly):
    """Get Axis-Aligned Bounding Box of a polygon."""
    return np.min(poly, axis=0), np.max(poly, axis=0)

def _aabb_overlap(aabb1, aabb2):
    """Check if two AABBs overlap."""
    min1, max1 = aabb1
    min2, max2 = aabb2
    if max1[0] < min2[0] or max2[0] < min1[0]:
        return False
    if max1[1] < min2[1] or max2[1] < min1[1]:
        return False
    return True

def _seg_dist(p1, p2, p3, p4):
    """Compute minimum distance between two line segments (p1-p2) and (p3-p4)."""
    p13 = p3 - p1
    p41 = p4 - p1
    d1 = p2 - p1
    d2 = p4 - p3
    d1_len2 = np.dot(d1, d1)
    d2_len2 = np.dot(d2, d2)
    
    # Handle degenerate segments
    if d1_len2 < 1e-10 and d2_len2 < 1e-10:
        return np.linalg.norm(p1 - p3)
    if d1_len2 < 1e-10:
        return np.linalg.norm(p3 - p1)
    if d2_len2 < 1e-10:
        return np.linalg.norm(p1 - p3)
        
    denom = d1_len2 * d2_len2 - np.dot(d1, d2) ** 2
    if denom < 1e-10:
        # Parallel
        s = 0.0
        t = np.dot(p13, d2) / d2_len2
        t = np.clip(t, 0.0, 1.0)
        s = (np.dot(d2, d2) * np.dot(p13, d1) - np.dot(d1, d2) * np.dot(p13, d2)) / denom
        s = np.clip(s, 0.0, 1.0)
    else:
        s = (np.dot(d2, d2) * np.dot(p13, d1) - np.dot(d1, d2) * np.dot(p13, d2)) / denom
        t = (d1_len2 * np.dot(p13, d2) - np.dot(d1, d2) * np.dot(p13, d1)) / denom
        s = np.clip(s, 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0)
        
    c1 = p1 + s * d1
    c2 = p3 + t * d2
    return np.linalg.norm(c1 - c2)

def _poly_distance(poly1, poly2):
    """
    Compute minimum distance between two convex polygons.
    Returns 0 if overlapping, positive if separated.
    Uses AABB pre-check and segment-segment distance.
    """
    # Quick AABB check
    aabb1 = _get_aabb(poly1)
    aabb2 = _get_aabb(poly2)
    if not _aabb_overlap(aabb1, aabb2):
        # Distance between corners of AABBs is a lower bound, but for simplicity 
        # if AABBs don't touch, they are separated. We need actual distance.
        # For disjoint convex polygons, min dist is min over all pairs of edges.
        pass
    
    min_dist = float('inf')
    n1 = len(poly1)
    n2 = len(poly2)
    
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            d = _seg_dist(p1, p2, p3, p4)
            if d < min_dist:
                min_dist = d
                if min_dist < 1e-9:
                    return 0.0
    return min_dist

def _is_valid_config(centers, angles, S, outer_angle_rad=0.0, tol=1e-4):
    """
    Check if configuration is valid:
    1. No overlaps between inner hexagons.
    2. All inner hexagons are inside the outer hexagon of side S.
    """
    polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    
    # 1. Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_distance(polys[i], polys[j])
            if d < tol:
                return False
            
    # 2. Check containment
    # Outer hexagon centered at (0,0) with side S and rotation outer_angle_rad.
    # A point is inside if its projection on all 6 outward normals is <= S * sqrt(3) / 2.
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    limit = S * math.sqrt(3) / 2.0
    
    for i in range(N):
        verts = polys[i]
        for k in range(6):
            n = outer_normals[k]
            projs = verts @ n
            if np.max(projs) > limit + tol or np.min(projs) < -limit - tol:
                return False
    return True

def _compute_min_side_length(centers, angles, outer_angle_rad=0.0):
    """
    Compute the minimum side length S such that all inner hexagons fit inside
    the outer hexagon centered at origin with rotation outer_angle_rad.
    """
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    max_abs_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        for k in range(6):
            n = outer_normals[k]
            projs = verts @ n
            max_abs_proj = max(max_abs_proj, np.max(np.abs(projs)))
            
    S = max_abs_proj * 2.0 / math.sqrt(3.0)
    return S

def _objective(params):
    """
    Objective function for Differential Evolution.
    params: 3N + 1 array [x1, y1, a1, ..., xN, yN, aN, outer_angle]
    """
    centers = np.column_stack([params[0::3], params[1::3]])
    angles = params[2::3]
    outer_angle = params[3*N]
    
    # Compute required side length
    S = _compute_min_side_length(centers, angles, outer_angle)
    
    # Check for overlaps
    polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _poly_distance(polys[i], polys[j])
            if d < 1e-4:
                # Heavy penalty for overlap
                overlap_penalty += (1e-4 - d) ** 2 * 1000.0
    
    # Add a small regularization to keep hexagons near center if possible?
    # No, just minimize S and penalize overlap.
    return S + overlap_penalty

def _seed_solution():
    """Generate a valid seed solution using a hexagonal lattice."""
    d = math.sqrt(3)  # Distance between centers of touching hexagons
    centers = []
    angles = []
    
    # Central hexagon
    centers.append([0, 0])
    angles.append(0.0)
    
    # 6 surrounding hexagons
    for i in range(6):
        theta = i * math.pi / 3.0
        cx = d * math.cos(theta)
        cy = d * math.sin(theta)
        centers.append([cx, cy])
        angles.append(0.0)
        
    # 4 more from second ring
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
                
    # Select 4 that are far apart
    selected_indices = [0, 3, 6, 9]
    for idx in selected_indices:
        if idx < len(ring2):
            centers.append(list(ring2[idx]))
            angles.append(0.0)
            
    centers = np.array(centers)
    angles = np.array(angles)
    
    S = _compute_min_side_length(centers, angles)
    return centers, angles, S

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    # Initialize with a valid seed
    centers0, angles0, S0 = _seed_solution()
    best_S = S0
    best_centers = centers0.copy()
    best_angles = angles0.copy()
    best_outer_angle = 0.0
    
    # Verify seed validity
    if not _is_valid_config(best_centers, best_angles, best_S, best_outer_angle):
        # If seed is invalid (shouldn't be, but just in case), expand it
        d = math.sqrt(3) * 1.5
        centers_fb = []
        angles_fb = []
        centers_fb.append([0, 0])
        angles_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            centers_fb.append([d * math.cos(theta), d * math.sin(theta)])
            angles_fb.append(0.0)
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            centers_fb.append([2.5*d * math.cos(theta), 2.5*d * math.sin(theta)])
            angles_fb.append(0.0)
        best_centers = np.array(centers_fb)
        best_angles = np.array(angles_fb)
        best_S = _compute_min_side_length(best_centers, best_angles)
        
    # Define bounds for Differential Evolution
    # Centers should be within a reasonable range
    # Max side length is likely around 3-4. So centers within [-3, 3]
    bounds_x = (-3.5, 3.5)
    bounds_y = (-3.5, 3.5)
    bounds_a = (0, 2 * math.pi)
    
    bounds = []
    for i in range(N):
        bounds.append(bounds_x)
        bounds.append(bounds_y)
        bounds.append(bounds_a)
    bounds.append((0, 2 * math.pi)) # Outer angle
    
    # Initial population centered around the seed
    x0 = np.zeros(3*N + 1)
    for i in range(N):
        x0[3*i] = best_centers[i][0]
        x0[3*i+1] = best_centers[i][1]
        x0[3*i+2] = best_angles[i]
    x0[3*N] = best_outer_angle

    # Run Differential Evolution
    # We use a wrapper to track best solution and stop early if time runs out
    global_best_S = best_S
    global_best_params = x0.copy()
    
    def callback(xk, convergence):
        nonlocal global_best_S, global_best_params
        # xk is the current best point
        centers = np.column_stack([xk[0::3], xk[1::3]])
        angles = xk[2::3]
        outer_angle = xk[3*N]
        
        S_val = _compute_min_side_length(centers, angles, outer_angle)
        
        # Check validity strictly
        if _is_valid_config(centers, angles, S_val, outer_angle):
            if S_val < global_best_S:
                global_best_S = S_val
                global_best_params = xk.copy()
                
        # Time check
        if time.time() > deadline - 5.0:
            raise Exception("Time limit reached")

    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            polish=True, # Use L-BFGS-B for local refinement
            callback=callback,
            update_interval=10,
            seed=42
        )
    except Exception as e:
        # If DE raises an exception due to time limit, we use global_best_params
        pass
    except Exception:
        pass

    # Check the result from DE
    if 'x' in locals() and res.x is not None:
        centers_de = np.column_stack([res.x[0::3], res.x[1::3]])
        angles_de = res.x[2::3]
        outer_angle_de = res.x[3*N]
        S_de = _compute_min_side_length(centers_de, angles_de, outer_angle_de)
        
        if _is_valid_config(centers_de, angles_de, S_de, outer_angle_de):
            if S_de < global_best_S:
                global_best_S = S_de
                global_best_params = res.x.copy()

    # Final local refinement using Nelder-Mead on the best found so far
    if time.time() < deadline - 5.0:
        # Use the best params found
        x_start = global_best_params.copy()
        
        def objective_local(params):
            centers = np.column_stack([params[0::3], params[1::3]])
            angles = params[2::3]
            outer_angle = params[3*N]
            S = _compute_min_side_length(centers, angles, outer_angle)
            
            # Strict penalty for invalidity
            polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = _poly_distance(polys[i], polys[j])
                    if d < 1e-4:
                        penalty += (1e-4 - d) ** 2 * 1000.0
            
            # Containment check penalty (should be 0 if S is computed correctly)
            return S + penalty

        try:
            from scipy.optimize import minimize
            res_local = minimize(
                objective_local, 
                x_start, 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000}
            )
            
            centers_loc = np.column_stack([res_local.x[0::3], res_local.x[1::3]])
            angles_loc = res_local.x[2::3]
            outer_angle_loc = res_local.x[3*N]
            S_loc = _compute_min_side_length(centers_loc, angles_loc, outer_angle_loc)
            
            if _is_valid_config(centers_loc, angles_loc, S_loc, outer_angle_loc):
                if S_loc < global_best_S:
                    global_best_S = S_loc
                    global_best_params = res_local.x.copy()
        except Exception:
            pass

    # Extract final best solution
    best_centers = np.column_stack([global_best_params[0::3], global_best_params[1::3]])
    best_angles = global_best_params[2::3]
    best_outer_angle = global_best_params[3*N]
    best_S = global_best_S

    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])
    
    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_angle)

# EVOLVE-BLOCK-END
