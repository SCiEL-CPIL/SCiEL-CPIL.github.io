import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    def compute_min_side(hexes):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all given hexagons.
        Uses both max and min projections to compute required half-width on each normal.
        """
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        all_verts = np.vstack(hexes)
        proj = all_verts @ normals.T
        
        max_proj = proj.max(axis=0)  # shape (6,)
        min_proj = proj.min(axis=0)  # shape (6,)
        
        # For each normal direction, the half-width needed is (max - min)/2
        # But since the outer hexagon is centered at origin, we need the center to be at (max+min)/2
        # Actually, the outer hexagon can be centered anywhere. The minimum side length
        # requires finding the center that minimizes the max over all 6 directions of the half-width.
        # For a regular hexagon with side s, the apothem is s*sqrt(3)/2.
        # The constraint is: for each normal n_k, (max_proj_k - min_proj_k)/2 <= s*sqrt(3)/2
        # So s >= max_k((max_proj_k - min_proj_k)) / sqrt(3)
        
        half_widths = (max_proj - min_proj) / 2.0
        s = np.max(half_widths) * 2.0 / np.sqrt(3.0)
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

    def compute_smooth_objective(params, beta=30.0):
        """
        Smooth objective for global search using soft penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side(hexes)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Strong quadratic penalty to prevent overlaps
                    penalty += overlap * overlap * 1000.0 + overlap * 100.0
                    
        return s_min + penalty

    def compute_objective(params):
        """
        Objective: min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        s_min = compute_min_side(hexes)
        
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
                    # Strong penalty
                    penalty += overlap * 1000.0
                    
        return s_min + penalty

    # --- Optimization ---
    
    # Bounds
    # r: 0 to 5
    # theta: 0 to 2pi
    # alpha: 0 to pi/3 (60 deg, due to hex symmetry)
    bounds = [
        (0.5, 5.0),  # r1
        (0.0, 2*np.pi), # theta1
        (0.0, np.pi/3), # alpha1
        (0.5, 5.0),  # r2
        (0.0, 2*np.pi), # theta2
        (0.0, np.pi/3), # alpha2
    ]
    
    best_s = np.inf
    best_params = None
    
    # Phase 1: Differential Evolution with smooth objective
    try:
        res_de = differential_evolution(
            lambda p: compute_smooth_objective(p, beta=30.0),
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False
        )
        if res_de.fun < best_s:
            best_s = res_de.fun
            best_params = res_de.x.copy()
    except Exception:
        pass

    # Phase 2: Basin Hopping
    if best_params is not None:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=30,
                T=0.5,
                stepsize=0.3,
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x.copy()
        except Exception:
            pass

    # Phase 3: Local Refinement with high precision
    if best_params is not None:
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
                best_params = res_local.x.copy()
        except Exception:
            pass

    # Phase 4: Random Restarts with better sampling
    if time.time() < deadline - 15:
        n_restarts = 15
        np.random.seed(123)
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Sample from a broader range with some structure
            r1 = np.random.uniform(1.0, 3.5)
            th1 = np.random.uniform(0, 2*np.pi)
            al1 = np.random.uniform(0, np.pi/3)
            r2 = np.random.uniform(1.5, 4.5)
            th2 = np.random.uniform(0, 2*np.pi)
            al2 = np.random.uniform(0, np.pi/3)
            
            p = np.array([r1, th1, al1, r2, th2, al2])
            
            try:
                res_local = minimize(
                    compute_objective,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_polish = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 200}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = np.array([1.5, 0.0, 0.0, 2.5, 0.0, 0.0])
        
    r1, th1, al1, r2, th2, al2 = best_params
    
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
        
        inner.append([float(x1), float(y1), float(np.degrees(a1))])
        inner.append([float(x2), float(y2), float(np.degrees(a2))])
        
    # Calculate final outer side length
    s_final = compute_min_side([hex_vertices(i[0], i[1], 1.0, np.radians(i[2])) for i in inner])
    
    outer_center = (0.0, 0.0)
    outer_side = float(s_final)
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
