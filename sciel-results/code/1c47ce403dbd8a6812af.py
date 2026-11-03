import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0  # Keep more margin for optimization

    # Hexagon geometry constants
    d = math.sqrt(3)  # distance between centers of adjacent unit hexagons

    # Generate centers: center + ring 1 (6) + ring 2 (12)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))

    # Ring 2: 12 hexagons
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))

    # Base centers (center + ring 1)
    base_centers = centers[:7]

    # Precompute vertices relative to center (unit hexagon, pointy-top)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    # Precompute base points (center + ring 1 vertices)
    base_points = []
    for cx, cy in base_centers:
        for vx, vy in verts_rel:
            base_points.append((cx + vx, cy + vy))
    base_points_arr = np.array(base_points)  # shape (42, 2)

    # Precompute ring 2 points for each of the 12 positions
    ring2_points_list = []
    for cx, cy in ring2_centers:
        pts = []
        for vx, vy in verts_rel:
            pts.append((cx + vx, cy + vy))
        ring2_points_list.append(np.array(pts))  # each shape (6, 2)

    # All combinations of 4 from 12 ring 2 positions
    all_combinations = list(itertools.combinations(range(12), 4))

    # Function to compute S for a given points array and theta
    def calc_S(points_arr, theta):
        # points_arr: (N_pts, 2)
        # theta: scalar
        k_angles = np.arange(6) * math.pi / 3
        angles = theta + k_angles
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (6, 2)
        
        # dots: (6, N_pts)
        dots = np.sum(normals[:, np.newaxis, :] * points_arr[np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        
        # max over points: (6,)
        max_per_normal = np.max(abs_dots, axis=-1)
        # max over normals: scalar
        max_proj = np.max(max_per_normal)
        
        return (2.0 / math.sqrt(3)) * max_proj

    best_S = float('inf')
    best_combo = None
    best_theta = 0.0
    top_candidates = [] # List of (S_coarse, combo)

    # Coarse sweep to identify top candidates
    n_coarse = 50 # Reduced to save time for optimization
    theta_coarse = np.linspace(0, math.pi / 3, n_coarse, endpoint=False)
    
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
        
        # Build points array for this combination
        combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])  # (66, 2)
        
        # Vectorized coarse sweep
        k_angles = np.arange(6) * math.pi / 3
        angles = theta_coarse[:, np.newaxis] + k_angles[np.newaxis, :]
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (100, 6, 2)
        
        # dots: (100, 6, 66)
        dots = np.sum(normals[:, :, np.newaxis, :] * combo_points[np.newaxis, np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        
        # max over points: (100, 6)
        max_per_normal = np.max(abs_dots, axis=-1)
        # max over normals: (100,)
        max_proj = np.max(max_per_normal, axis=-1)
        S_vals = (2.0 / math.sqrt(3)) * max_proj
        
        min_idx = np.argmin(S_vals)
        min_S = S_vals[min_idx]
        min_theta = theta_coarse[min_idx]
        
        top_candidates.append((min_S, combo, min_theta))
        
        if min_S < best_S:
            best_S = min_S
            best_combo = combo
            best_theta = min_theta

    # Sort candidates by S
    top_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 5 candidates using scipy.optimize.minimize_scalar
    refined_best_S = float('inf')
    refined_best_combo = None
    refined_best_theta = 0.0
    
    if time.time() < deadline - safety_margin:
        for S_coarse, combo, theta_init in top_candidates[:5]:
            if time.time() > deadline - safety_margin:
                break
                
            combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])
            
            # Define objective function for minimize_scalar
            # It expects a scalar x
            def obj(theta):
                return calc_S(combo_points, theta)
                
            # Use bounded method for reliability
            # The function is periodic with period pi/3
            # We can search in [0, pi/3]
            try:
                res = minimize_scalar(obj, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-6})
                if res.success:
                    S_opt = res.fun
                    theta_opt = res.x
                    if S_opt < refined_best_S:
                        refined_best_S = S_opt
                        refined_best_combo = combo
                        refined_best_theta = theta_opt
            except Exception:
                continue

    # If refinement improved, use it
    if refined_best_combo is not None:
        best_S = refined_best_S
        best_combo = refined_best_combo
        best_theta = refined_best_theta

    # Fallback
    if best_combo is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Build final center list for the rigid body solution
    final_centers = list(base_centers)
    for idx in best_combo:
        final_centers.append(ring2_centers[idx])
    
    # --- LOCAL OPTIMIZATION TO REDUCE SIDE LENGTH ---
    # We allow the 11 hexagons to move and rotate to fit in a smaller hexagon.
    # We start from the best rigid configuration.
    
    if time.time() < deadline - safety_margin:
        # Initial parameters: x, y, angle for each of the 11 hexagons
        # We fix the outer hexagon to angle 0 and center (0,0) to reduce variables?
        # Actually, we can just optimize the inner positions. The outer side S is derived from the max projection.
        
        init_params = []
        for cx, cy in final_centers:
            init_params.extend([cx, cy, 0.0])
        
        init_params = np.array(init_params)
        
        def objective(params):
            # params: (33,)
            # Unpack
            xs = params[0::3]
            ys = params[1::3]
            thetas = params[2::3]
            
            # Check for overlap (penalty)
            # Distance between centers must be >= sqrt(3)
            penalty = 0.0
            for i in range(11):
                for j in range(i+1, 11):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 3.0: # sqrt(3)^2
                        # Penalty proportional to overlap
                        # dist < sqrt(3)
                        dist = math.sqrt(dist_sq)
                        penalty += (math.sqrt(3) - dist) ** 2 * 10.0 # High weight
            
            # Calculate vertices for all hexagons
            # We want to minimize the max projection on 6 normals (angle 0)
            # Normals for hexagon with angle 0:
            # k_angles = [0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3]
            # Normals are perpendicular to sides.
            # For a regular hexagon with vertices at angles 0, pi/3..., the normals are at pi/6, pi/2... ?
            # Let's stick to the projection definition used in calc_S.
            # calc_S uses normals at theta + k*pi/3.
            # If outer hexagon is fixed at angle 0, normals are at 0, pi/3...
            
            max_proj = 0.0
            for i in range(11):
                # Rotate vertices
                c = math.cos(thetas[i])
                s = math.sin(thetas[i])
                # verts_rel is (6,2)
                # Rotated verts:
                rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
                ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
                
                # Translate
                px = rx + xs[i]
                py = ry + ys[i]
                
                # Project on 6 normals
                # Normals:
                for k in range(6):
                    ang = k * math.pi / 3
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                    proj = np.max(np.abs(px * nx + py * ny))
                    if proj > max_proj:
                        max_proj = proj
            
            # Side length S = 2 * max_proj / sqrt(3)
            S = (2.0 / math.sqrt(3)) * max_proj
            
            return S + penalty
        
        try:
            # Use Nelder-Mead
            # We might need to restrict the search space or use bounds, but NM doesn't support bounds directly.
            # We can rely on the penalty to keep them close.
            res = minimize(objective, init_params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success:
                final_params = res.x
                xs = final_params[0::3]
                ys = final_params[1::3]
                thetas = final_params[2::3]
                
                # Recalculate S without penalty to get the true side length
                # We must ensure no overlap for validity.
                # The penalty method might result in slight overlaps if not converged.
                # Let's check validity.
                valid = True
                for i in range(11):
                    for j in range(i+1, 11):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        if dx*dx + dy*dy < 3.0 - 1e-5: # Allow tiny numerical error
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Calculate S
                    max_proj = 0.0
                    for i in range(11):
                        c = math.cos(thetas[i])
                        s = math.sin(thetas[i])
                        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
                        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
                        px = rx + xs[i]
                        py = ry + ys[i]
                        for k in range(6):
                            ang = k * math.pi / 3
                            nx = math.cos(ang)
                            ny = math.sin(ang)
                            proj = np.max(np.abs(px * nx + py * ny))
                            if proj > max_proj:
                                max_proj = proj
                    new_S = (2.0 / math.sqrt(3)) * max_proj
                    
                    if new_S < best_S:
                        best_S = new_S
                        # Update final_centers and angles for output
                        final_centers = []
                        final_angles = []
                        for i in range(11):
                            final_centers.append((xs[i], ys[i]))
                            final_angles.append(math.degrees(thetas[i]))
                        # We need to return inner rows with angles
                        # The current return format is:
                        # inner: [[x, y, angle_deg], ...]
                        # center: (x, y)
                        # side: S
                        # angle: outer_angle
                        # We assumed outer angle 0.
                        best_outer_angle = 0.0
                        best_inner_data = [[xs[i], ys[i], math.degrees(thetas[i])] for i in range(11)]
        except Exception:
            pass
    # --- END LOCAL OPTIMIZATION ---

    # If local optimization didn't run or failed, use the rigid body result
    if 'best_inner_data' in locals() and best_inner_data is not None:
        return best_inner_data, (0.0, 0.0), best_S, best_outer_angle

    # Format output for rigid body
    inner_rows = []
    for cx, cy in final_centers:
        inner_rows.append([cx, cy, 0.0])

    center = (0.0, 0.0)
    side = best_S
    angle_deg = math.degrees(best_theta)

    return inner_rows, center, side, angle_deg
