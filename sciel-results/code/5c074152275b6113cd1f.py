import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time
import math

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

    def sat_overlap(v1, v2, normals1, normals2):
        """
        Returns min overlap depth (negative if disjoint) using SAT with precomputed normals.
        """
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
        # Normals for a hexagon rotated by outer_angle.
        base_angles = np.radians([30, 90, 150, 210, 270, 330])
        normal_angles = base_angles + outer_angle
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def get_hexes_and_normals_from_params(params):
        """
        params: [r1, theta1, alpha1, r2, theta2, alpha2]
        Returns list of 12 hexagons (6x2 arrays) and their normals.
        Uses 6-fold rotational symmetry.
        Outer hexagon is fixed at angle 0 for optimization stability, 
        but we return the angle 0.0 for the final call if we fix it.
        Actually, let's keep outer_angle as a parameter to be safe, but 
        we can fix it to 0 in the optimization to reduce dims.
        Let's stick to 6 params: r1, t1, a1, r2, t2, a2.
        """
        r1, t1, a1, r2, t2, a2 = params
        
        # Convert polar to cartesian for the base positions
        x1 = r1 * np.cos(t1)
        y1 = r1 * np.sin(t1)
        x2 = r2 * np.cos(t2)
        y2 = r2 * np.sin(t2)
        
        hexes = []
        normals_list = []
        
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            
            # Rotate position 1
            x1_rot = x1 * c - y1 * s_
            y1_rot = x1 * s_ + y1 * c
            a1_rot = a1 + angle
            
            hex1 = hex_vertices(x1_rot, y1_rot, 1.0, a1_rot)
            hexes.append(hex1)
            # Precompute normals for hex1
            base_norm_angles = np.radians([30, 90, 150, 210, 270, 330]) + a1_rot
            normals_list.append(np.column_stack([np.cos(base_norm_angles), np.sin(base_norm_angles)]))
            
            # Rotate position 2
            x2_rot = x2 * c - y2 * s_
            y2_rot = x2 * s_ + y2 * c
            a2_rot = a2 + angle
            
            hex2 = hex_vertices(x2_rot, y2_rot, 1.0, a2_rot)
            hexes.append(hex2)
            base_norm_angles2 = np.radians([30, 90, 150, 210, 270, 330]) + a2_rot
            normals_list.append(np.column_stack([np.cos(base_norm_angles2), np.sin(base_norm_angles2)]))
            
        return hexes, normals_list

    def compute_objective(params, smooth=False):
        """
        Objective: min_side + penalty for overlaps.
        params: [r1, theta1, alpha1, r2, theta2, alpha2]
        Outer angle is fixed at 0.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes, normals_list = get_hexes_and_normals_from_params(params)
        
        # Outer angle is 0
        s_min = compute_min_side(hexes, 0.0)
        
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
                    
                # Fast SAT with precomputed normals
                min_overlap = sat_overlap(hexes[i], hexes[j], normals_list[i], normals_list[j])
                
                if min_overlap > 0:
                    if smooth:
                        # Smooth penalty for global search
                        penalty += np.log(1.0 + np.exp(10.0 * min_overlap))
                    else:
                        # Linear penalty for local search
                        penalty += min_overlap * 10.0
                        
        return s_min + penalty

    # --- Optimization ---
    
    # Bounds
    # r: 0 to 5
    # theta: 0 to pi/3 (60 deg)
    # alpha: 0 to pi/3 (60 deg)
    bounds = [
        (0.0, 5.0),  # r1
        (0.0, np.pi/3), # theta1
        (0.0, np.pi/3), # alpha1
        (0.0, 5.0),  # r2
        (0.0, np.pi/3), # theta2
        (0.0, np.pi/3), # alpha2
    ]
    
    best_s = np.inf
    best_params = None
    
    # Initial guess: Honeycomb-like
    # Hex 1: r=1.5, theta=0, alpha=0
    # Hex 2: r=3.0, theta=0, alpha=0
    initial_guess = np.array([
        1.5, 0.0, 0.0,  # r1, t1, a1
        3.0, 0.0, 0.0   # r2, t2, a2
    ])
    
    # Phase 1: Local refinement from honeycomb initial guess
    try:
        res_local_init = minimize(
            lambda p: compute_objective(p, smooth=False),
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        if res_local_init.fun < best_s:
            best_s = res_local_init.fun
            best_params = res_local_init.x
    except Exception:
        pass
    
    # Phase 2: Differential Evolution
    if time.time() < deadline - 30:
        try:
            res_de = differential_evolution(
                lambda p: compute_objective(p, smooth=True),
                bounds,
                maxiter=300,
                popsize=25,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if res_de.fun < best_s:
                best_s = res_de.fun
                best_params = res_de.x
        except Exception:
            pass

    # Phase 3: Basin Hopping
    if best_params is not None and time.time() < deadline - 20:
        try:
            def local_minimize(x):
                res = minimize(
                    lambda p: compute_objective(p, smooth=False),
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=30,
                T=0.5,
                stepsize=0.3,
                minimizer_kwargs={'options': {'maxiter': 80}},
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x
        except Exception:
            pass

    # Phase 4: Local Refinement
    if best_params is not None and time.time() < deadline - 10:
        try:
            res_local = minimize(
                lambda p: compute_objective(p, smooth=False),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x
        except Exception:
            pass

    # Phase 5: Random Restarts with time budget
    if time.time() < deadline - 10:
        time_per_restart = 5.0
        n_restarts = int((deadline - time.time() - 5) / time_per_restart)
        n_restarts = max(1, min(n_restarts, 15))
        
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            p = np.array([
                np.random.uniform(0.5, 4.0), np.random.uniform(0, np.pi/3), np.random.uniform(0, np.pi/3),
                np.random.uniform(1.0, 4.5), np.random.uniform(0, np.pi/3), np.random.uniform(0, np.pi/3)
            ])
            
            try:
                res_local = minimize(
                    lambda q: compute_objective(q, smooth=False),
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass

    # Fallback
    if best_params is None:
        best_params = initial_guess
        
    r1, t1, a1, r2, t2, a2 = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    x1 = r1 * np.cos(t1)
    y1 = r1 * np.sin(t1)
    x2 = r2 * np.cos(t2)
    y2 = r2 * np.sin(t2)
    
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        # Rotate base 1
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        # Rotate base 2
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([x1_rot, y1_rot, np.degrees(a1_rot)])
        inner.append([x2_rot, y2_rot, np.degrees(a2_rot)])
        
    # Calculate final outer side length
    hexes_final = [hex_vertices(i[0], i[1], 1.0, np.radians(i[2])) for i in inner]
    s_final = compute_min_side(hexes_final, 0.0)
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            res_polish = minimize(
                lambda p: compute_objective(p, smooth=False),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x
                r1, t1, a1, r2, t2, a2 = best_params
                
                # Regenerate inner hexagons
                x1 = r1 * np.cos(t1)
                y1 = r1 * np.sin(t1)
                x2 = r2 * np.cos(t2)
                y2 = r2 * np.sin(t2)
                
                inner = []
                for k in range(6):
                    angle = np.radians(60 * k)
                    c = np.cos(angle)
                    s_ = np.sin(angle)
                    
                    x1_rot = x1 * c - y1 * s_
                    y1_rot = x1 * s_ + y1 * c
                    a1_rot = a1 + angle
                    
                    x2_rot = x2 * c - y2 * s_
                    y2_rot = x2 * s_ + y2 * c
                    a2_rot = a2 + angle
                    
                    inner.append([x1_rot, y1_rot, np.degrees(a1_rot)])
                    inner.append([x2_rot, y2_rot, np.degrees(a2_rot)])
                
                hexes_final = [hex_vertices(i[0], i[1], 1.0, np.radians(i[2])) for i in inner]
                s_final = compute_min_side(hexes_final, 0.0)
        except Exception:
            pass
    
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
