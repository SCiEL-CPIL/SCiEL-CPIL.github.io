from scipy.optimize import basinhopping
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

    def hex_vertex_normals(angle_rad):
        """Returns 6x2 array of outward unit normals for a regular hexagon with given orientation."""
        angles = angle_rad + np.radians([30, 90, 150, 210, 270, 330])
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2, n1=None, n2=None):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
        if n1 is None:
            n1 = get_normals(v1)
        if n2 is None:
            n2 = get_normals(v2)
        
        normals = np.vstack([n1, n2])
        
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

    def get_hexes_from_params(params):
        """
        params: [r1, theta1, alpha1, r2, theta2, alpha2]
        Returns list of 12 hexagons (6x2 arrays).
        """
        r1, th1, al1, r2, th2, al2 = params
        
        hexes = []
        # Orbit 1
        for k in range(6):
            angle = np.radians(60 * k)
            # Position
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            # Rotation: base rotation + angle of position
            a1 = al1 + angle
            hexes.append(hex_vertices(x1, y1, 1.0, a1))
            
        # Orbit 2
        for k in range(6):
            angle = np.radians(60 * k)
            # Position
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            # Rotation
            a2 = al2 + angle
            hexes.append(hex_vertices(x2, y2, 1.0, a2))
            
        return hexes

    def get_hexes_and_normals_from_params(params):
        """
        Returns list of (vertices, normals) tuples for 12 hexagons.
        """
        r1, th1, al1, r2, th2, al2 = params
        
        hexes = []
        normals_list = []
        
        for k in range(6):
            angle = np.radians(60 * k)
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            a1 = al1 + angle
            hexes.append(hex_vertices(x1, y1, 1.0, a1))
            normals_list.append(hex_vertex_normals(a1))
            
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            a2 = al2 + angle
            hexes.append(hex_vertices(x2, y2, 1.0, a2))
            normals_list.append(hex_vertex_normals(a2))
            
        return hexes, normals_list

    def compute_objective(params):
        """
        Objective: min_side + penalty for overlaps.
        params: [r1, theta1, alpha1, r2, theta2, alpha2, outer_angle]
        """
        if time.time() > deadline:
            return 1e10
            
        # Extract outer angle
        outer_angle = params[6] if len(params) > 6 else 0.0
        
        hexes, normals_list = get_hexes_and_normals_from_params(params[:6])
        
        s_min = compute_min_side(hexes, outer_angle)
        
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
                normals = np.vstack([normals_list[i], normals_list[j]])
                p1 = hexes[i] @ normals.T
                p2 = hexes[j] @ normals.T
                min1p = p1.min(axis=0)
                max1p = p1.max(axis=0)
                min2p = p2.min(axis=0)
                max2p = p2.max(axis=0)
                overlaps = np.minimum(max1p, max2p) - np.maximum(min1p, min2p)
                min_overlap = np.min(overlaps)
                
                if min_overlap > 0:
                    penalty += min_overlap * 10.0
                    
        return s_min + penalty

    def compute_smooth_objective(params):
        """
        Smooth objective for global search.
        """
        if time.time() > deadline:
            return 1e10
            
        outer_angle = params[6] if len(params) > 6 else 0.0
        hexes, normals_list = get_hexes_and_normals_from_params(params[:6])
        
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
                    
                normals = np.vstack([normals_list[i], normals_list[j]])
                p1 = hexes[i] @ normals.T
                p2 = hexes[j] @ normals.T
                min1p = p1.min(axis=0)
                max1p = p1.max(axis=0)
                min2p = p2.min(axis=0)
                max2p = p2.max(axis=0)
                overlaps = np.minimum(max1p, max2p) - np.maximum(min1p, min2p)
                min_overlap = np.min(overlaps)
                
                if min_overlap > 0:
                    # Smooth penalty
                    penalty += np.log(1.0 + np.exp(5.0 * min_overlap)) * 1.0
                    
        return s_min + penalty

    # --- Optimization ---
    
    # Bounds
    # r: 0 to 5
    # theta: 0 to 2pi
    # alpha: 0 to pi/3 (60 deg, due to hex symmetry)
    # outer_angle: 0 to pi/3 (60 deg symmetry of outer hexagon)
    bounds = [
        (0.5, 5.0),  # r1
        (0.0, 2*np.pi), # theta1
        (0.0, np.pi/3), # alpha1
        (0.5, 5.0),  # r2
        (0.0, 2*np.pi), # theta2
        (0.0, np.pi/3), # alpha2
        (0.0, np.pi/3), # outer_angle
    ]
    
    best_s = np.inf
    best_params = None
    
    # Honeycomb-like initial guess
    # For 12 hexagons in a hexagonal arrangement:
    # Inner ring of 6 at radius ~1.8, outer ring of 6 at radius ~3.6
    # Angles aligned for honeycomb packing
    initial_guess = np.array([
        1.8, 0.0, 0.0,  # r1, theta1, alpha1
        3.6, 0.0, 0.0,  # r2, theta2, alpha2
        0.0             # outer_angle
    ])
    
    # Phase 1: Local refinement from honeycomb initial guess
    try:
        res_local_init = minimize(
            compute_objective,
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
                compute_smooth_objective,
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
                    compute_objective,
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
                T=0.8,
                stepsize=0.4,
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
                compute_objective,
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
                np.random.uniform(1, 3), np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi/3),
                np.random.uniform(2, 4), np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi/3),
                np.random.uniform(0, np.pi/3)
            ])
            
            try:
                res_local = minimize(
                    compute_objective,
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
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        try:
            # Try to further refine
            res_polish = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x
                r1, th1, al1, r2, th2, al2, outer_angle_rad = best_params
                
                # Regenerate inner hexagons
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
                
                hexes_final = [hex_vertices(i[0], i[1], 1.0, np.radians(i[2])) for i in inner]
                s_final = compute_min_side(hexes_final, outer_angle_rad)
        except Exception:
            pass
    
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = np.degrees(outer_angle_rad)

    return inner, outer_center, outer_side, outer_angle_deg
