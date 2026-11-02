import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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
        # Normals for a hexagon with flat top/bottom (or pointy top/bottom)
        # The side length S is related to the projection onto normals by:
        # projection <= S * sqrt(3) / 2
        # So S = 2 / sqrt(3) * max(projection)
        
        # Let's use normals at 0, 60, 120, 180, 240, 300 degrees
        # This corresponds to a hexagon with vertices at 30, 90, ... 
        # Wait, if normals are at 0, 60..., the vertices are at 30, 90...
        # The distance from center to side is apothem = S * sqrt(3) / 2.
        # So max projection onto normal = S * sqrt(3) / 2.
        
        normal_angles = np.radians([0, 60, 120, 180, 240, 300])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_projs = []
        for i in range(6):
            n = normals[i]
            max_proj = -np.inf
            for h in hexes:
                projs = h @ n
                max_proj = max(max_proj, np.max(projs))
            max_projs.append(max_proj)
        
        max_proj = np.max(max_projs)
        # S * sqrt(3) / 2 = max_proj
        # S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3.0)

    def compute_objective_symmetric(params):
        """
        params: [x1, y1, a1, x2, y2, a2, x3, y3, a3, x4, y4, a4]
        Generates 12 hexagons using 3-fold symmetry.
        Returns min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        # 4 base hexagons
        base_hexes = []
        for k in range(4):
            x = params[k*3]
            y = params[k*3+1]
            a = params[k*3+2]
            base_hexes.append(hex_vertices(x, y, 1.0, a))
        
        # Generate 12 hexagons by 3-fold symmetry
        hexes = []
        for k in range(3):
            angle = np.radians(120 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            for bh in base_hexes:
                h = bh @ rot.T
                hexes.append(h)
        
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
                    # Smooth penalty
                    eps = 1e-6
                    smooth_pen = 0.5 * (overlap + np.sqrt(overlap**2 + eps))
                    penalty += smooth_pen * 10.0
                    
        return min_s + penalty

    # --- Optimization ---
    
    bounds = []
    for _ in range(4):
        bounds.append((-4.0, 4.0))  # x
        bounds.append((-4.0, 4.0))  # y
        bounds.append((0.0, np.pi/3)) # angle

    best_s = np.inf
    best_params = None
    
    # Use Differential Evolution for global search
    try:
        res = differential_evolution(
            compute_objective_symmetric,
            bounds,
            maxiter=100, # Reduced to allow more polishing
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=False # We will do our own polishing
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception as e:
        pass
    
    # Polish with Nelder-Mead
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
                noise = np.random.uniform(-0.05, 0.05, size=12)
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
        # Try random starts
        for _ in range(20):
            p = np.array([
                np.random.uniform(1, 3), np.random.uniform(-1, 1), np.random.uniform(0, np.pi/3),
                np.random.uniform(1, 3), np.random.uniform(-1, 1), np.random.uniform(0, np.pi/3),
                np.random.uniform(1, 3), np.random.uniform(-1, 1), np.random.uniform(0, np.pi/3),
                np.random.uniform(1, 3), np.random.uniform(-1, 1), np.random.uniform(0, np.pi/3)
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
                
    # Generate the 12 inner hexagons
    inner = []
    for k in range(3):
        angle = np.radians(120 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        for m in range(4):
            x = best_params[m*3]
            y = best_params[m*3+1]
            a = best_params[m*3+2]
            
            # Rotate (x, y)
            x_rot = x * c - y * s_
            y_rot = x * s_ + y * c
            a_rot = a + angle
            
            inner.append([x_rot, y_rot, np.degrees(a_rot)])
            
    # Compute the actual minimum outer side length
    # Reconstruct hexes to compute min side
    base_hexes = []
    for m in range(4):
        x = best_params[m*3]
        y = best_params[m*3+1]
        a = best_params[m*3+2]
        base_hexes.append(hex_vertices(x, y, 1.0, a))
        
    hexes = []
    for k in range(3):
        angle = np.radians(120 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        rot = np.array([[c, -s_], [s_, c]])
        
        for bh in base_hexes:
            h = bh @ rot.T
            hexes.append(h)
    
    # Use the exact computation for the final score
    outer_side = compute_min_side_exact(hexes)
    
    outer_center = (0.0, 0.0)
    outer_angle_deg = 0.0
    
    return inner, outer_center, outer_side, outer_angle_deg
