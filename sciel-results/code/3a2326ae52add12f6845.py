"""Differentiable packing strategy using L-BFGS-B and smooth loss functions."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    SQRT3 = math.sqrt(3)
    
    # Vertices of a unit hexagon centered at origin, with 0 rotation
    # Vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # Wait, standard "pointy top" or "flat top"?
    # The task says "regular hexagons". Orientation is defined by angle_degrees.
    # Let's use vertices at angles 0, 60, ..., 300 relative to the hexagon's local frame.
    # If angle_degrees = 0, vertex 0 is at (1, 0).
    base_vertices = np.array([
        [math.cos(math.radians(i*60)), math.sin(math.radians(i*60))] for i in range(6)
    ])
    
    # Helper: Compute vertices of all hexagons given parameters
    # params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_angle]
    # a is in radians for internal computation, but input is degrees? 
    # The return format requires angle_degrees. 
    # It is better to optimize in radians for smoothness, then convert.
    def get_all_hex_vertices(params):
        # params shape: (N*3 + 1, )
        # Last element is outer angle in radians
        outer_angle = params[-1]
        hexagons = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            # Rotation matrix
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            verts = base_vertices @ rot.T + np.array([x, y])
            hexagons.append(verts)
        return hexagons, outer_angle

    # Helper: Compute support function for a set of hexagons in a given direction
    # We need to find the minimum outer hexagon side length S such that all hexagons fit.
    # The outer hexagon is centered at (0,0) for simplicity (we can center the inner hexes later).
    # Actually, the outer hexagon center is also a variable? 
    # The interface allows (inner, center, side, angle).
    # To simplify, we can fix the outer center to (0,0) by centering the inner hexagons' centroid at (0,0) in the loss.
    # Or better, treat outer center as (0,0) and optimize inner positions.
    # The outer hexagon has 6 edges. The normal to edge k is at angle outer_angle + 30 + k*60.
    # The apothem of the outer hexagon is S * SQRT3 / 2.
    # For a point p to be inside the outer hexagon, its projection onto every outward normal n must be <= apothem.
    # So, max_k max_i (p_i . n_k) <= S * SQRT3 / 2.
    # Thus S >= (2 / SQRT3) * max_k max_i (p_i . n_k).
    
    def calc_side_length(params):
        hexagons, outer_angle = get_all_hex_vertices(params)
        # Normals of outer hexagon
        # Edge k normal is perpendicular to edge. 
        # If vertices are at 0, 60..., the edge between v0 and v1 has normal at 30 deg.
        # So normals are at outer_angle + 30 + k*60.
        max_proj = 0.0
        for k in range(6):
            theta = math.radians(30 + k*60) + outer_angle
            nx = math.cos(theta)
            ny = math.sin(theta)
            # Project all vertices of all hexagons onto this normal
            for hex_verts in hexagons:
                projections = hex_verts[:, 0] * nx + hex_verts[:, 1] * ny
                if np.max(projections) > max_proj:
                    max_proj = np.max(projections)
        
        # S * SQRT3 / 2 >= max_proj  =>  S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # Helper: Compute smooth overlap penalty
    # Two convex polygons P and Q overlap if for some axis, their projections overlap.
    # They are disjoint if there exists an axis where projections are separated.
    # Distance between two convex polygons can be approximated by the maximum separation over all axes.
    # Let d(P, Q) = max_axis (min_proj_Q - max_proj_P) if separated, else negative (penetration).
    # Actually, we want a penalty if they are NOT separated.
    # A common smooth approximation for "minimum distance" or "separation":
    # For each axis, compute gap = min_Q - max_P. If gap > 0, they are separated along this axis.
    # If they are disjoint, max_axis(gap) > 0.
    # If they overlap, max_axis(gap) < 0.
    # We want to penalize when max_axis(gap) < 0.
    # Let m = max_axis(gap). 
    # Penalty = softplus(-m) * weight. Or simply max(0, -m)^2.
    # To make it differentiable, we can use a smooth max of the gaps.
    # However, the number of axes is 12 (6 from P, 6 from Q).
    
    def get_axes_for_hex(a):
        # Normals to edges of a hexagon with rotation a
        axes = []
        for k in range(6):
            theta = math.radians(30 + k*60) + a
            axes.append((math.cos(theta), math.sin(theta)))
        return axes

    def smooth_max(values, beta=10.0):
        # Log-sum-exp approximation for max
        # values is a list of scalars
        v = np.array(values)
        v_max = np.max(v)
        return v_max + math.log(np.sum(np.exp((v - v_max) * beta))) / beta

    def overlap_penalty(params):
        hexagons, _ = get_all_hex_vertices(params)
        # Precompute axes for each hexagon
        axes_list = []
        for i in range(N):
            a = params[i*3+2]
            axes_list.append(get_axes_for_hex(a))
        
        total_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                p1 = hexagons[i]
                p2 = hexagons[j]
                
                # All axes to check: axes of p1 and axes of p2
                all_axes = axes_list[i] + axes_list[j]
                
                gaps = []
                for nx, ny in all_axes:
                    proj1 = p1[:, 0] * nx + p1[:, 1] * ny
                    proj2 = p2[:, 0] * nx + p2[:, 1] * ny
                    
                    min1 = np.min(proj1)
                    max1 = np.max(proj1)
                    min2 = np.min(proj2)
                    max2 = np.max(proj2)
                    
                    # Gap is the distance between the two intervals.
                    # If max1 < min2, gap = min2 - max1 (positive)
                    # If max2 < min1, gap = min1 - max2 (positive)
                    # If they overlap, gap is negative (penetration depth)
                    # We want the "separation" along this axis.
                    # sep = min(min2 - max1, min1 - max2) ?? No.
                    # The distance between two intervals [a,b] and [c,d] is:
                    # max(0, max(a-c, c-b)) is not quite right for overlap.
                    # If [a,b] and [c,d] overlap, the "gap" is negative.
                    # Let's define gap = min(min2 - max1, min1 - max2).
                    # If separated, one of these is positive and the other is very negative?
                    # Example: P=[0,1], Q=[2,3]. min2-max1 = 2-1=1. min1-max2 = 0-3=-3.
                    # We want the positive gap. So gap = max(min2 - max1, min1 - max2).
                    # If P=[0,1], Q=[0.5, 1.5]. min2-max1 = 0.5-1=-0.5. min1-max2 = 0-1.5=-1.5.
                    # max is -0.5. This represents the penetration depth.
                    gap = max(min2 - max1, min1 - max2)
                    gaps.append(gap)
                
                # If the hexagons are disjoint, there exists an axis where gap > 0.
                # So max(gaps) > 0.
                # If they overlap, max(gaps) < 0.
                # We want to penalize when max(gaps) < 0.
                # Let m = smooth_max(gaps).
                # If m < 0, penalty = -m.
                # To be smooth, penalty = softplus(-m) or max(0, -m)^2.
                # Using smooth max for m makes the function differentiable almost everywhere.
                m = smooth_max(gaps, beta=50.0)
                if m < 0:
                    total_penalty += (-m) ** 2
                else:
                    # Small penalty if barely separated to encourage robust separation?
                    # No, 0 is fine.
                    pass
        
        return total_penalty

    # Objective function
    def objective(params):
        # Center the configuration to keep it near origin
        # Calculate centroid of inner hexagons
        cx_avg = np.mean(params[0:N*3:3])
        cy_avg = np.mean(params[1:N*3:3])
        
        # Shifted params for evaluation (centering)
        centered_params = params.copy()
        for i in range(N):
            centered_params[i*3] -= cx_avg
            centered_params[i*3+1] -= cy_avg
        
        side = calc_side_length(centered_params)
        pen = overlap_penalty(centered_params)
        
        # Weight for penalty
        # If side is ~2, penalty should be significant if overlap.
        # Overlap penalty is squared penetration. Penetration ~0.1 => 0.01.
        # We want penalty to dominate if overlap is significant.
        # Let's use a large weight.
        weight = 100.0
        return side + weight * pen

    # Gradient calculation? 
    # L-BFGS-B can use numerical gradients or analytical.
    # Numerical gradients are slow but easier to implement correctly.
    # Given N=11, 33+1=34 variables. Numerical gradient takes 34 evaluations.
    # Each evaluation involves N^2 overlap checks (55 pairs) and side calc.
    # This might be slow. Let's try to optimize or use fewer iterations.
    # Alternatively, use 'BFGS' or 'Nelder-Mead' if gradients are too slow.
    # But the directive asks for a differentiable strategy. 
    # Let's implement a simple numerical gradient and see if it fits in time.
    # Or use scipy's finite difference which is optimized.
    
    # Initial parameters
    # Start with a hexagonal lattice arrangement
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Generate a few random initializations
    best_result = None
    best_score = float('inf')
    
    # Configuration 1: Compact lattice
    coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    # Configuration 2: Random scatter
    random.seed(42)
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if restart == 0:
            # Use lattice
            params = []
            for q, r in coords1:
                x, y = axial_to_cartesian(q, r)
                params.extend([x, y, 0.0])
            params.append(0.0) # outer angle
        else:
            # Random perturbation of lattice or random start
            params = []
            for q, r in coords1:
                x, y = axial_to_cartesian(q, r)
                # Add noise
                x += random.gauss(0, 0.5)
                y += random.gauss(0, 0.5)
                a = random.gauss(0, 0.5)
                params.extend([x, y, a])
            params.append(random.gauss(0, 0.5))
        
        params = np.array(params)
        
        # Optimize
        # Using L-BFGS-B with numerical gradient
        # To save time, we can limit maxiter
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-5, 'maxls': 20})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception as e:
            # If L-BFGS fails, try Nelder-Mead as fallback
            try:
                res = minimize(objective, params, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
            except:
                pass

    # If no valid result found, fallback to lattice
    if best_result is None:
        params = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r)
            params.extend([x, y, 0.0])
        params.append(0.0)
        best_result = np.array(params)
        best_score = objective(best_result)

    # Final validation and extraction
    final_params = best_result
    # Center the final result
    cx_avg = np.mean(final_params[0:N*3:3])
    cy_avg = np.mean(final_params[1:N*3:3])
    
    inner_list = []
    for i in range(N):
        x = final_params[i*3] - cx_avg
        y = final_params[i*3+1] - cy_avg
        a_rad = final_params[i*3+2]
        a_deg = math.degrees(a_rad) % 360.0
        inner_list.append([x, y, a_deg])
    
    outer_angle_deg = math.degrees(final_params[-1]) % 360.0
    
    # Calculate final side length
    # We need to recalculate the side length for the centered configuration
    # and ensure the outer hexagon is centered at (0,0)
    
    # Re-evaluate side length with centered params
    centered_params = final_params.copy()
    for i in range(N):
        centered_params[i*3] -= cx_avg
        centered_params[i*3+1] -= cy_avg
        
    side = calc_side_length(centered_params)
    
    # Verify disjointness (hard check)
    # If invalid, we should return a valid configuration.
    # The optimization should have minimized overlap, but it's possible it didn't reach 0.
    # Let's do a quick check.
    hexagons, _ = get_all_hex_vertices(centered_params)
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            p1 = hexagons[i]
            p2 = hexagons[j]
            # Simple check: min distance > 0
            # Use SAT for robustness
            a1 = centered_params[i*3+2]
            a2 = centered_params[j*3+2]
            axes1 = get_axes_for_hex(a1)
            axes2 = get_axes_for_hex(a2)
            all_axes = axes1 + axes2
            sep = False
            for nx, ny in all_axes:
                proj1 = p1[:, 0] * nx + p1[:, 1] * ny
                proj2 = p2[:, 0] * nx + p2[:, 1] * ny
                if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                    sep = True
                    break
            if not sep:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to pure lattice
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r)
            inner_list.append([x, y, 0.0])
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, 0.0] for x, y, a in inner_list]
        
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        side = calc_outer_side_fast(hexagons, 0.0)
        return (inner_list, (0.0, 0.0), side, 0.0)

    return (inner_list, (0.0, 0.0), side, outer_angle_deg)

# Helper functions used in fallback (copied from seed for safety)
def get_hex_vertices(cx, cy, angle_deg):
    verts = []
    for i in range(6):
        theta = math.radians(angle_deg + i * 60)
        x = cx + math.cos(theta)
        y = cy + math.sin(theta)
        verts.append((x, y))
    return verts

def calc_outer_side_fast(hexagons, outer_angle_deg):
    SQRT3 = math.sqrt(3)
    base_angles = [30, 90, 150, 210, 270, 330]
    axis_dirs = []
    for a in base_angles:
        rad = math.radians(a + outer_angle_deg)
        axis_dirs.append((math.cos(rad), math.sin(rad)))
    
    max_proj = 0.0
    for hex_verts in hexagons:
        for vx, vy in hex_verts:
            for nx, ny in axis_dirs:
                proj = abs(vx * nx + vy * ny)
                if proj > max_proj:
                    max_proj = proj
    return 2.0 * max_proj / SQRT3

# EVOLVE-BLOCK-END
