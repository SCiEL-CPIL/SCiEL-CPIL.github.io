import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
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
        
        # Collect all projections for each normal direction
        # We need the max projection for each of the 6 normals
        max_projs = []
        for i in range(6):
            n = normals[i]
            projs = []
            for h in hexes:
                p = h @ n
                projs.append(np.max(p))
            max_projs.append(np.max(projs))
        
        max_projs = np.array(max_projs)
        
        # The side length is determined by the maximum of these 6 projections
        # s * sqrt(3) / 2 = max(max_projs)
        # So s = max(max_projs) * 2 / sqrt(3)
        
        return np.max(max_projs) * 2.0 / np.sqrt(3.0)

    def compute_objective_symmetric(params):
        """
        params: [x1, y1, a1, x2, y2, a2]
        Generates 12 hexagons using 6-fold symmetry.
        Returns min_side + penalty for overlaps.
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
            
            # Rotate base1 and base2
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        # Compute minimum outer side length (exact)
        min_s = compute_min_side_exact(hexes)
        
        penalty = 0.0
        
        # Check overlaps
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
                    # Smooth penalty: smooth_plus(overlap)
                    # smooth_plus(x) = 0.5 * (x + sqrt(x^2 + eps))
                    eps = 1e-6
                    smooth_pen = 0.5 * (overlap + np.sqrt(overlap**2 + eps))
                    penalty += smooth_pen * 10.0
                    
        return min_s + penalty

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
    
    # Use Differential Evolution for global search
    try:
        res = differential_evolution(
            compute_objective_symmetric,
            bounds,
            maxiter=200, # Reduced to allow more polishing
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=True
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Polish with Nelder-Mead (handles non-smooth exact objective better than L-BFGS-B)
    if best_params is not None:
        try:
            res_nm = minimize(
                compute_objective_symmetric,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            # Check if the new solution is within bounds
            if np.all(res_nm.x >= np.array([b[0] for b in bounds]) - 1e-9) and np.all(res_nm.x <= np.array([b[1] for b in bounds]) + 1e-9):
                if res_nm.fun < best_s:
                    best_s = res_nm.fun
                    best_params = res_nm.x
        except Exception:
            pass
        
        # Try a few local restarts around the best found to escape potential local minima
        if time.time() < deadline - 20:
            for restart in range(5):
                # Add small noise
                noise = np.random.uniform(-0.05, 0.05, size=6)
                p_restarted = best_params + noise
                # Clip to bounds
                p_restarted = np.clip(p_restarted, [b[0] for b in bounds], [b[1] for b in bounds])
                
                try:
                    res_r = minimize(
                        compute_objective_symmetric,
                        p_restarted,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if np.all(res_r.x >= np.array([b[0] for b in bounds]) - 1e-9) and np.all(res_r.x <= np.array([b[1] for b in bounds]) + 1e-9):
                        if res_r.fun < best_s:
                            best_s = res_r.fun
                            best_params = res_r.x
                except Exception:
                    pass
                
                if time.time() > deadline - 10:
                    break

    # Construct the final answer from best_params
    if best_params is None:
        # Fallback: Simple grid or known approximate
        for _ in range(10):
            p = np.array([
                np.random.uniform(1, 3), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3),
                np.random.uniform(2, 4), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3)
            ])
            val = compute_objective_symmetric(p)
            if val < best_s:
                best_s = val
                best_params = p
            try:
                res_local = minimize(
                    compute_objective_symmetric,
                    p,
                    method='Nelder-Mead',
                    options={'maxiter': 100}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass
            if time.time() > deadline - 5:
                break
                
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
        
    # Compute the actual minimum outer side length
    # Reconstruct hexes to compute min side
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
    
    # Use the exact computation for the final score
    outer_side = compute_min_side_exact(hexes)
    
    outer_center = (0.0, 0.0)
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
