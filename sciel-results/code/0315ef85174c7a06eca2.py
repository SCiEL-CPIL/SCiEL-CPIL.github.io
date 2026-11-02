import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return min_overlap

    def compute_min_side_exact(hexes):
        """
        Calculate the exact minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all given hexagons.
        """
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def soft_max(x, beta):
        """
        Stable soft-max approximation of max(x).
        x: array of values
        beta: sharpness parameter
        """
        # Shift to prevent overflow
        x_max = np.max(x)
        if x_max > 0:
            x_shifted = x - x_max
        else:
            x_shifted = x
            
        # Clip to avoid underflow issues in exp for very negative numbers, 
        # though exp(-large) is just 0 which is fine.
        # However, if x is very large positive, we handled it with shift.
        exp_vals = np.exp(beta * x_shifted)
        sum_exp = np.sum(exp_vals)
        
        if sum_exp == 0:
            return x_max
            
        return x_max + (1.0 / beta) * np.log(sum_exp)

    def soft_plus(x, beta):
        """
        Smooth approximation of max(0, x).
        """
        if x <= 0:
            # For very negative x, softplus is ~0
            # Using the formula: (1/beta) * log(1 + exp(beta*x))
            # If beta*x is very negative, exp is 0, log(1)=0.
            # To be safe and fast:
            if x < -10.0 / beta:
                return 0.0
        return (1.0 / beta) * np.log(1.0 + np.exp(beta * x))

    def compute_objective_smooth(params, beta_sm=50.0, beta_sp=20.0):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        Returns smooth min_side + smooth penalty.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate the 12 hexagons
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        # 1. Smooth Outer Side Length
        # Collect all projections of all vertices onto the 6 outer normals
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        all_projs = []
        for h in hexes:
            proj = h @ normals.T
            all_projs.append(proj.flatten())
            
        all_projs = np.concatenate(all_projs)
        
        # Use soft-max to approximate the max projection
        # The max projection determines the side length.
        # s = (2/sqrt(3)) * max_proj
        max_proj_smooth = soft_max(all_projs, beta_sm)
        s_smooth = max_proj_smooth * 2.0 / np.sqrt(3.0)
        
        # 2. Smooth Overlap Penalty
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check to avoid SAT if far apart
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                # If overlap is negative (disjoint), softplus should be ~0
                # If overlap is positive, softplus approximates overlap
                penalty += soft_plus(overlap, beta_sp)
                
        return s_smooth + penalty * 10.0

    def compute_objective_exact(params):
        """
        Exact objective for final verification or fallback.
        """
        if time.time() > deadline:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        s_min = compute_min_side_exact(hexes)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 1000.0 # Heavy penalty for exact
                
        return s_min + penalty

    # --- Optimization ---
    
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
    ]
    
    best_s = np.inf
    best_params = None
    
    # Phase 1: Global Search with Smooth Objective
    # Using a moderate beta for smoothness to help DE and polish
    try:
        res = differential_evolution(
            lambda p: compute_objective_smooth(p, beta_sm=30.0, beta_sp=15.0),
            bounds,
            maxiter=150,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=True,
            polish_method=lambda x, fun: minimize(fun, x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x.copy()
            
    except Exception as e:
        pass
    
    # Phase 2: Local Refinement with Smoother Objective
    # Increase beta to sharpen the approximation towards the exact max
    if best_params is not None:
        try:
            res_local = minimize(
                lambda p: compute_objective_smooth(p, beta_sm=100.0, beta_sp=50.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
            
    # Phase 3: Final Polish with Exact Objective (if time permits)
    # The smooth objective converges to a point very close to the exact optimum.
    # We can try a few steps of L-BFGS-B on the exact objective, but it's non-smooth.
    # Instead, we just evaluate the exact score of the best_params found.
    # If there is still time, we can do a very small local search around best_params
    # using the exact objective with a very small step size (simulated annealing or just random jitter + minimize).
    
    if time.time() < deadline - 10 and best_params is not None:
        # Try a few random perturbations and local minimize on exact objective
        # This helps if the soft-max approximation left a tiny residual error
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Small jitter
            jitter = np.random.normal(0, 0.01, size=6)
            # Ensure angles stay in bounds roughly
            p_try = best_params + jitter
            # Clip angles
            p_try[2] = np.clip(p_try[2], 0, np.pi/3)
            p_try[5] = np.clip(p_try[5], 0, np.pi/3)
            
            # Evaluate exact
            val_exact = compute_objective_exact(p_try)
            
            if val_exact < best_s:
                best_s = val_exact
                best_params = p_try.copy()
                
            # Try to minimize from this point using smooth objective again?
            # Or just accept if it's good.
            # Let's try a quick minimize on smooth objective from this new point
            try:
                res_fine = minimize(
                    lambda p: compute_objective_smooth(p, beta_sm=200.0, beta_sp=100.0),
                    p_try,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 20, 'ftol': 1e-15}
                )
                if res_fine.fun < best_s:
                    # Check exact score of this refined point
                    val_check = compute_objective_exact(res_fine.x)
                    if val_check < best_s:
                        best_s = val_check
                        best_params = res_fine.x.copy()
            except:
                pass

    # Construct the final answer from best_params
    if best_params is None:
        # Fallback: Simple grid or known approximate
        # This should rarely happen
        best_params = np.array([2.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        
    x1, y1, a1, x2, y2, a2 = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        # Rotate (x1, y1)
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        # Rotate (x2, y2)
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([x1_rot, y1_rot, np.degrees(a1_rot)])
        inner.append([x2_rot, y2_rot, np.degrees(a2_rot)])
        
    # Calculate final outer side length EXACTLY
    s_final = compute_min_side_exact([hex_vertices(i[0], i[1], 1.0, np.radians(i[2])) for i in inner])
    
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
