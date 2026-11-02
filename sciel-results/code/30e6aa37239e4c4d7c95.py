import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s kill, leave 20s)

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
        """Returns min overlap depth (negative if disjoint) using SAT."""
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

    def get_rigid_cluster(offset_x, offset_y, rotation):
        """
        Returns 12 hexagons (6x2 arrays) in a rigid honeycomb cluster:
        - 1 center hexagon at origin
        - 6 hexagons in ring 1 (distance 2, angles 0,60,...,300)
        - 5 hexagons in ring 2 (distance 2*sqrt(3), angles 30,90,150,210,270)
        All rotated by `rotation` and translated by (offset_x, offset_y).
        """
        hexes = []
        
        # Center hexagon
        hexes.append(hex_vertices(offset_x, offset_y, 1.0, rotation))
        
        # Ring 1: 6 hexagons at distance 2
        for k in range(6):
            angle = np.radians(60 * k)
            x1 = 2.0 * np.cos(angle)
            y1 = 2.0 * np.sin(angle)
            # Apply global rotation
            xr = x1 * np.cos(rotation) - y1 * np.sin(rotation)
            yr = x1 * np.sin(rotation) + y1 * np.cos(rotation)
            hexes.append(hex_vertices(offset_x + xr, offset_y + yr, 1.0, rotation))
        
        # Ring 2: 5 hexagons at distance 2*sqrt(3), angles 30,90,150,210,270
        dist2 = 2.0 * np.sqrt(3.0)
        for k in range(5):
            angle = np.radians(30 + 60 * k)
            x2 = dist2 * np.cos(angle)
            y2 = dist2 * np.sin(angle)
            # Apply global rotation
            xr = x2 * np.cos(rotation) - y2 * np.sin(rotation)
            yr = x2 * np.sin(rotation) + y2 * np.cos(rotation)
            hexes.append(hex_vertices(offset_x + xr, offset_y + yr, 1.0, rotation))
        
        return hexes

    def compute_objective_rigid(params):
        """
        Objective for rigid cluster: params = [offset_x, offset_y, rotation, outer_angle]
        Returns min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        ox, oy, rot, outer_angle = params
        hexes = get_rigid_cluster(ox, oy, rot)
        
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
                    penalty += overlap * 10.0
                    
        return s_min + penalty

    def compute_smooth_objective_rigid(params):
        """
        Smooth objective for global search.
        """
        if time.time() > deadline:
            return 1e10
            
        ox, oy, rot, outer_angle = params
        hexes = get_rigid_cluster(ox, oy, rot)
        
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
                    penalty += np.log(1.0 + np.exp(5.0 * overlap)) * 1.0
                    
        return s_min + penalty

    # --- Optimization ---
    
    # Bounds for rigid cluster
    # offset: -3 to 3
    # rotation: 0 to pi/3 (60 deg symmetry)
    # outer_angle: 0 to pi/3 (60 deg symmetry)
    bounds = [
        (-3.0, 3.0),  # offset_x
        (-3.0, 3.0),  # offset_y
        (0.0, np.pi/3), # rotation
        (0.0, np.pi/3), # outer_angle
    ]
    
    best_s = np.inf
    best_params = None
    
    # Phase 1: Differential Evolution
    try:
        res_de = differential_evolution(
            compute_smooth_objective_rigid,
            bounds,
            maxiter=300,
            popsize=25,
            tol=1e-8,
            seed=42,
            polish=True
        )
        if res_de.fun < best_s:
            best_s = res_de.fun
            best_params = res_de.x
    except Exception:
        pass

    # Phase 2: Basin Hopping
    if best_params is not None:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective_rigid,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                return res.fun, res.x, {}
                
            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=40,
                T=0.5,
                stepsize=0.3,
                minimizer_kwargs={'options': {'maxiter': 100}},
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x
        except Exception:
            pass

    # Phase 3: Local Refinement
    if best_params is not None:
        try:
            res_local = minimize(
                compute_objective_rigid,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x
        except Exception:
            pass

    # Phase 4: Random Restarts
    if time.time() < deadline - 10:
        n_restarts = 15
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            p = np.array([
                np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                np.random.uniform(0, np.pi/3),
                np.random.uniform(0, np.pi/3)
            ])
            
            try:
                res_local = minimize(
                    compute_objective_rigid,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass

    # Fallback
    if best_params is None:
        best_params = np.array([0.0, 0.0, 0.0, 0.0])
        
    ox, oy, rot, outer_angle_rad = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    hexes_final = get_rigid_cluster(ox, oy, rot)
    for h in hexes_final:
        # Extract center and angle from the hexagon
        cx = np.mean(h[:, 0])
        cy = np.mean(h[:, 1])
        # Determine angle from first vertex
        vx = h[0, 0] - cx
        vy = h[0, 1] - cy
        a1 = np.arctan2(vy, vx)
        inner.append([cx, cy, np.degrees(a1)])
        
    # Calculate final outer side length
    s_final = compute_min_side(hexes_final, outer_angle_rad)
    
    outer_center = (0.0, 0.0)
    outer_side = s_final
    outer_angle_deg = np.degrees(outer_angle_rad)

    return inner, outer_center, outer_side, outer_angle_deg
