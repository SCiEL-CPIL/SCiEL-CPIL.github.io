import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin

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

    def compute_min_side(hexes, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, orientation outer_angle)
        that contains all given hexagons.
        """
        base_angles = np.radians([30, 90, 150, 210, 270, 330])
        normal_angles = base_angles + outer_angle
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def smooth_max(values, temperature=0.05):
        """
        Smooth approximation of max using log-sum-exp.
        """
        if len(values) == 0:
            return 0.0
        v_max = np.max(values)
        if v_max == -np.inf:
            return -np.inf
        shifted = (values - v_max) / temperature
        shifted = np.clip(shifted, -50, 50)
        result = v_max + temperature * np.log(np.sum(np.exp(shifted)))
        return result

    def get_hexes_from_params(params, outer_angle=0.0):
        """
        Returns list of 12 hexagons (vertices) from 6 parameters: [r1, th1, al1, r2, th2, al2]
        Assumes 6-fold symmetry.
        """
        r1, th1, al1, r2, th2, al2 = params
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            a1 = al1 + angle
            hexes.append(hex_vertices(x1, y1, 1.0, a1))
            
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            a2 = al2 + angle
            hexes.append(hex_vertices(x2, y2, 1.0, a2))
            
        return hexes

    def check_disjoint(hexes, tol=-1e-5):
        """
        Returns True if all hexagons are disjoint (overlap < tol).
        """
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > tol:
                    return False
        return True

    def compute_objective_smooth(params, temperature=0.05):
        """
        Objective: smooth min_side + penalty for overlaps.
        params: [r1, theta1, alpha1, r2, theta2, alpha2, outer_angle]
        """
        if time.time() > deadline:
            return 1e10
            
        outer_angle = params[6] if len(params) > 6 else 0.0
        hexes = get_hexes_from_params(params[:6], outer_angle)
        
        # Compute smooth min side
        base_angles = np.radians([30, 90, 150, 210, 270, 330])
        normal_angles = base_angles + outer_angle
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        all_projs = []
        for h in hexes:
            proj = h @ normals.T
            all_projs.extend(proj.flatten())
            
        max_proj_smooth = smooth_max(np.array(all_projs), temperature)
        s_min_smooth = max_proj_smooth * 2.0 / np.sqrt(3.0)
        
        penalty = 0.0
        # Check overlaps
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
                    # Use softplus for smooth penalty
                    penalty += np.log(1 + np.exp(overlap / temperature)) * temperature * 100.0
                    
        return s_min_smooth + penalty

    def compute_objective_exact(params):
        """
        Objective: exact min_side + penalty for overlaps.
        params: [r1, theta1, alpha1, r2, theta2, alpha2, outer_angle]
        """
        if time.time() > deadline:
            return 1e10
            
        outer_angle = params[6] if len(params) > 6 else 0.0
        hexes = get_hexes_from_params(params[:6], outer_angle)
        
        s_min = compute_min_side(hexes, outer_angle)
        
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
                    penalty += overlap * 100.0
                    
        return s_min + penalty

    # --- Optimization ---
    
    bounds = [
        (0.5, 5.0),  # r1
        (0.0, 2*np.pi), # theta1
        (0.0, np.pi/3), # alpha1
        (0.5, 5.0),  # r2
        (0.0, 2*np.pi), # theta2
        (0.0, np.pi/3), # alpha2
        (0.0, np.pi/3), # outer_angle
    ]
    
    best_s_valid = np.inf
    best_params_valid = None
    best_s_any = np.inf
    best_params_any = None

    # Initial guesses
    # 1. Honeycomb-like (tight)
    init1 = np.array([1.8, 0.0, 0.0, 3.6, 0.0, 0.0, 0.0])
    # 2. Tighter ring
    init2 = np.array([1.732, 0.0, 0.0, 3.464, 0.0, 0.0, 0.0])
    # 3. Rotated inner ring
    init3 = np.array([1.8, 0.0, np.pi/6, 3.6, 0.0, 0.0, 0.0])
    
    initial_guesses = [init1, init2, init3]

    # Phase 1: Local refinement from multiple initial guesses using smooth objective
    for ig in initial_guesses:
        if time.time() > deadline - 10: break
        try:
            res_local_init = minimize(
                compute_objective_smooth,
                ig,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res_local_init.fun < best_s_any:
                best_s_any = res_local_init.fun
                best_params_any = res_local_init.x
                
            # Check validity with exact objective
            hexes = get_hexes_from_params(res_local_init.x[:6], res_local_init.x[6])
            if check_disjoint(hexes):
                s_val = compute_min_side(hexes, res_local_init.x[6])
                if s_val < best_s_valid:
                    best_s_valid = s_val
                    best_params_valid = res_local_init.x
        except Exception:
            pass
    
    # Phase 1b: Adaptive temperature refinement with tighter smooth objective
    if best_params_any is not None and time.time() < deadline - 15:
        for temp in [0.01, 0.005]:
            if time.time() > deadline - 5: break
            try:
                def obj_tight(params, temp=temp):
                    return compute_objective_smooth(params, temperature=temp)
                
                res_tight = minimize(
                    obj_tight,
                    best_params_any,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                if res_tight.fun < best_s_any:
                    best_s_any = res_tight.fun
                    best_params_any = res_tight.x
                    
                hexes = get_hexes_from_params(res_tight.x[:6], res_tight.x[6])
                if check_disjoint(hexes):
                    s_val = compute_min_side(hexes, res_tight.x[6])
                    if s_val < best_s_valid:
                        best_s_valid = s_val
                        best_params_valid = res_tight.x
            except Exception:
                pass

    # Phase 2: Differential Evolution using smooth objective
    if time.time() < deadline - 30:
        try:
            res_de = differential_evolution(
                compute_objective_smooth,
                bounds,
                maxiter=200,
                popsize=25,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if res_de.fun < best_s_any:
                best_s_any = res_de.fun
                best_params_any = res_de.x
                
            hexes = get_hexes_from_params(res_de.x[:6], res_de.x[6])
            if check_disjoint(hexes):
                s_val = compute_min_side(hexes, res_de.x[6])
                if s_val < best_s_valid:
                    best_s_valid = s_val
                    best_params_valid = res_de.x
        except Exception:
            pass

    # Phase 3: Basin Hopping using smooth objective
    if best_params_any is not None and time.time() < deadline - 20:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective_smooth,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params_any,
                niter=30,
                T=0.8,
                stepsize=0.4,
                minimizer_kwargs={'options': {'maxiter': 80}},
                seed=42
            )
            if res_bh.fun < best_s_any:
                best_s_any = res_bh.fun
                best_params_any = res_bh.x
                
            hexes = get_hexes_from_params(res_bh.x[:6], res_bh.x[6])
            if check_disjoint(hexes):
                s_val = compute_min_side(hexes, res_bh.x[6])
                if s_val < best_s_valid:
                    best_s_valid = s_val
                    best_params_valid = res_bh.x
        except Exception:
            pass

    # Phase 4: Random Restarts using smooth objective
    if time.time() < deadline - 10:
        time_per_restart = 5.0
        n_restarts = int((deadline - time.time() - 5) / time_per_restart)
        n_restarts = max(1, min(n_restarts, 15))
        
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            p = np.array([
                np.random.uniform(1, 3), np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi/3),
                np.random.uniform(2, 4), np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi/3),
                np.random.uniform(0, np.pi/3)
            ])
            
            try:
                res_local = minimize(
                    compute_objective_smooth,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_local.fun < best_s_any:
                    best_s_any = res_local.fun
                    best_params_any = res_local.x
                    
                hexes = get_hexes_from_params(res_local.x[:6], res_local.x[6])
                if check_disjoint(hexes):
                    s_val = compute_min_side(hexes, res_local.x[6])
                    if s_val < best_s_valid:
                        best_s_valid = s_val
                        best_params_valid = res_local.x
            except Exception:
                pass

    # Use best valid if available, else best any
    if best_params_valid is not None:
        best_params = best_params_valid
    else:
        best_params = best_params_any
        if best_params is None:
            best_params = initial_guesses[0]

    r1, th1, al1, r2, th2, al2, outer_angle_rad = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        
        x1 = r1 * np.cos(th1 + angle)
        y1 = r1 * np.sin(th1 + angle)
        a1 = al1 + angle
        
        x2 = r2 * np.cos(th2 + angle)
        y2 = r2 * np.sin(th2 + angle)
        a2 = al2 + angle
        
        inner.append([x1, y1, np.degrees(a1)])
        inner.append([x2, y2, np.degrees(a2)])
        
    # Calculate final outer side length
    hexes_final = [hex_vertices(i[0], i[1], 1.0, np.radians(i[2])) for i in inner]
    s_final = compute_min_side(hexes_final, outer_angle_rad)
    
    # Final high-precision polish with exact objective
    if time.time() < deadline - 2:
        try:
            res_polish = minimize(
                compute_objective_exact,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            # Only update if valid
            hexes_p = get_hexes_from_params(res_polish.x[:6], res_polish.x[6])
            if check_disjoint(hexes_p):
                s_p = compute_min_side(hexes_p, res_polish.x[6])
                if s_p < s_final:
                    s_final = s_p
                    best_params = res_polish.x
                    r1, th1, al1, r2, th2, al2, outer_angle_rad = best_params
                    inner = []
                    for k in range(6):
                        angle = np.radians(60 * k)
                        x1 = r1 * np.cos(th1 + angle)
                        y1 = r1 * np.sin(th1 + angle)
                        a1 = al1 + angle
                        x2 = r2 * np.cos(th2 + angle)
                        y2 = r2 * np.sin(th2 + angle)
                        a2 = al2 + angle
                        inner.append([x1, y1, np.degrees(a1)])
                        inner.append([x2, y2, np.degrees(a2)])
        except Exception:
            pass
    
    # Final micro-polish with very tight smooth objective
    if time.time() < deadline - 1:
        try:
            def obj_micro(params):
                return compute_objective_smooth(params, temperature=0.001)
            
            res_micro = minimize(
                obj_micro,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            hexes_m = get_hexes_from_params(res_micro.x[:6], res_micro.x[6])
            if check_disjoint(hexes_m):
                s_m = compute_min_side(hexes_m, res_micro.x[6])
                if s_m < s_final:
                    s_final = s_m
                    best_params = res_micro.x
                    r1, th1, al1, r2, th2, al2, outer_angle_rad = best_params
                    inner = []
                    for k in range(6):
                        angle = np.radians(60 * k)
                        x1 = r1 * np.cos(th1 + angle)
                        y1 = r1 * np.sin(th1 + angle)
                        a1 = al1 + angle
                        x2 = r2 * np.cos(th2 + angle)
                        y2 = r2 * np.sin(th2 + angle)
                        a2 = al2 + angle
                        inner.append([x1, y1, np.degrees(a1)])
                        inner.append([x2, y2, np.degrees(a2)])
        except Exception:
            pass
    
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = np.degrees(outer_angle_rad)

    return inner, outer_center, outer_side, outer_angle_deg
