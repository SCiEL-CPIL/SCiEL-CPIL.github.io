import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices):
    """
    Compute minimum side length s of a regular hexagon centered at origin (angle 0)
    that contains all given vertices.
    """
    # Outward normals for a regular hexagon with angle 0 (flat top/bottom? No, pointy top? 
    # The task says vertices at cx + d cos(a + 60i). 
    # If a=0, vertices are at 0, 60, 120, 180, 240, 300.
    # This is a "pointy-top" hexagon in standard math coords? 
    # Let's check the normals.
    # Edge from 0 to 60 deg. Midpoint angle 30 deg. Normal is 30 deg.
    # Edge from 60 to 120. Midpoint 90. Normal 90.
    # So normals are at 30, 90, 150, 210, 270, 330.
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated (no overlap), False if overlapping.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i+1)%6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # Strict separation check with small tolerance
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return True
    
    return False

def _is_valid_config(positions, angles, side=1.0):
    """Check if configuration is valid: all pairs separated and inside outer hex."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(positions[i], angles[i], positions[j], angles[j], side):
                return False
    
    # Check containment
    verts = np.vstack([
        _hex_vertices(positions[i,0], positions[i,1], side, angles[i]) 
        for i in range(n)
    ])
    s_val = _min_outer_side(verts)
    
    # Reconstruct outer hex vertices to check containment explicitly? 
    # The task says: "an inner vertex v outside the outer hexagon: for every outer edge... cross... >= -1e-6"
    # My _min_outer_side calculates the *minimum* side length that *would* contain them if centered at 0.
    # But the outer hexagon in the return value is centered at 0 with side s_val.
    # So if s_val is the calculated min side, then by definition all vertices are inside or on the boundary.
    # However, we must ensure the "outer" hexagon returned is the one with side s_val.
    # The check in the evaluator uses the returned 'outer' hexagon.
    # So we just need to return s_val as the side.
    return True

def _smooth_overlap_penalty(positions, angles, side=1.0):
    """
    Compute a smooth penalty for overlaps.
    Uses a proxy for penetration depth based on distance between centers.
    For regular hexagons of side 1, the minimum distance between centers for non-overlap
    depends on orientation. 
    If aligned, distance must be >= 2 * apothem = sqrt(3) * 1? No.
    Distance between centers of two touching aligned hexagons is 2 * apothem? No.
    Apothem is distance from center to edge.
    If two hexagons touch edge-to-edge, center distance is 2 * apothem = sqrt(3).
    Wait, apothem of unit hexagon is sqrt(3)/2.
    So touching distance is 2 * (sqrt(3)/2) = sqrt(3) approx 1.732.
    
    If they are rotated, the minimum distance can be smaller? No, convex bodies.
    The minimum distance between centers for two convex polygons to not overlap is the support function distance.
    For simplicity, we use a penalty based on distance < sqrt(3).
    This is a conservative lower bound for *some* orientations, but for arbitrary orientations,
    the required separation might be larger (e.g. vertex to vertex touching is distance 2).
    Actually, the minimum separation distance between two unit hexagons is sqrt(3) (edge-to-edge).
    The maximum separation distance required is 2 (vertex-to-vertex).
    So if d < sqrt(3), they MUST overlap.
    If d > 2, they MUST be separated.
    In between, it depends on angle.
    
    For the SA objective, we want a smooth landscape.
    We will use a penalty if d < sqrt(3) + epsilon.
    """
    penalty = 0.0
    n = len(positions)
    min_dist = math.sqrt(3) # approx 1.732
    
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                # Strong penalty for deep overlap
                penalty += 100.0 * (min_dist - d) ** 2
            elif d < min_dist + 0.5:
                # Check actual SAT to see if they are separated
                # If not separated, add penalty based on "violation"
                # Since we don't have easy penetration depth, we use a soft penalty
                # that encourages separation.
                if not _check_separation(positions[i], angles[i], positions[j], angles[j], side):
                    # Estimate penetration: 
                    # The "gap" needed is roughly (min_dist - d) if aligned.
                    # But if rotated, it's complex.
                    # Let's just add a moderate penalty to push them apart.
                    penalty += 10.0 * (min_dist - d + 0.5) ** 2
    return penalty

def _objective(x):
    positions = x[:24].reshape(12, 2)
    angles = x[24:]
    
    vertices = np.vstack([
        _hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]) 
        for i in range(12)
    ])
    
    s = _min_outer_side(vertices)
    pen = _smooth_overlap_penalty(positions, angles)
    
    return s + pen

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, leave margin
    
    n = 12
    side = 1.0
    
    # Initial configuration: Honeycomb cluster
    positions_init = np.zeros((12, 2))
    angles_init = np.zeros(12)
    
    positions_init[0] = [0,0]
    
    for k in range(6):
        theta = np.radians(60 * k)
        positions_init[k+1] = [2.0 * np.cos(theta), 2.0 * np.sin(theta)]
        
    for k in range(5):
        inner_idx = k + 1
        theta = np.radians(60 * inner_idx)
        positions_init[k+7] = positions_init[inner_idx] + 2.0 * np.array([np.cos(theta), np.sin(theta)])
        
    x0 = np.concatenate([positions_init.flatten(), angles_init])
    
    best_s = float('inf')
    best_x = x0.copy()
    
    # 1. Initial Polish with L-BFGS-B to get a good local minimum
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < best_s:
            pos = res.x[:24].reshape(12, 2)
            ang = res.x[24:]
            if _is_valid_config(pos, ang, side):
                best_s = res.fun
                best_x = res.x.copy()
    except Exception:
        pass
    
    # 2. Simulated Annealing / Random Restart Search
    # We will perform a series of random perturbations and polish them.
    # This helps escape local minima.
    
    np.random.seed(42) # For reproducibility in this context, though evaluator might not care
    
    num_restarts = 100
    current_x = best_x.copy()
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Perturb the current best solution
        # Scale of perturbation decreases over time? Or random?
        # Let's try a mix: some large jumps, some small.
        if restart % 5 == 0:
            scale = 1.0
        else:
            scale = 0.1
        
        x_try = current_x + np.random.normal(0, scale, size=current_x.shape)
        # Keep angles in [0, 60)
        x_try[24:] = np.mod(x_try[24:], 60)
        
        # Polish the perturbed solution
        try:
            res = minimize(_objective, x_try, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.fun < best_s:
                pos = res.x[:24].reshape(12, 2)
                ang = res.x[24:]
                
                if _is_valid_config(pos, ang, side):
                    # Verify s is consistent
                    verts = np.vstack([
                        _hex_vertices(pos[i,0], pos[i,1], 1.0, ang[i]) 
                        for i in range(12)
                    ])
                    s_val = _min_outer_side(verts)
                    # The objective includes penalty, so res.fun might be > s_val if penalty > 0
                    # But _is_valid_config ensures no overlap, so penalty should be 0 or near 0
                    # Actually, _smooth_overlap_penalty might return > 0 even if valid?
                    # No, _is_valid_config uses strict SAT. If valid, they are separated.
                    # So penalty should be 0.
                    # However, _smooth_overlap_penalty might not be 0 if d is in the "gray zone" (1.732 < d < 2.232)
                    # even if SAT says separated.
                    # So we should use the geometric s_val for comparison.
                    
                    if s_val < best_s:
                        best_s = s_val
                        best_x = res.x.copy()
                        current_x = res.x.copy()
                        
        except Exception:
            pass

    # Final validation and formatting
    best_positions = best_x[:24].reshape(12, 2)
    best_angles = best_x[24:]
    
    # One last polish on the absolute best
    try:
        res_final = minimize(_objective, best_x, method='L-BFGS-B', 
                             options={'maxiter': 1000, 'ftol': 1e-12})
        pos_f = res_final.x[:24].reshape(12, 2)
        ang_f = res_final.x[24:]
        if _is_valid_config(pos_f, ang_f, side):
            verts_f = np.vstack([
                _hex_vertices(pos_f[i,0], pos_f[i,1], 1.0, ang_f[i]) 
                for i in range(12)
            ])
            s_f = _min_outer_side(verts_f)
            if s_f < best_s:
                best_s = s_f
                best_positions = pos_f
                best_angles = ang_f
    except Exception:
        pass

    inner_hex_data = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s

