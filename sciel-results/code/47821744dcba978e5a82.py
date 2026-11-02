import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # --- Geometry Helpers ---
    
    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon (CCW)."""
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
        return float(np.min(overlaps))

    def smooth_max(values, temperature=0.01):
        """Smooth approximation of max using log-sum-exp."""
        if len(values) == 0:
            return 0.0
        v_max = np.max(values)
        if v_max == -np.inf:
            return -np.inf
        shifted = (values - v_max) / temperature
        shifted = np.clip(shifted, -50, 50)
        result = v_max + temperature * np.log(np.sum(np.exp(shifted)))
        return result

    def get_hexes_from_params(params):
        """
        params: [r1, theta1, alpha1, r2, theta2, alpha2]
        Returns list of 12 hexagons (6x2 arrays) using 6-fold symmetry.
        """
        r1, th1, al1, r2, th2, al2 = params
        
        hexes = []
        # Orbit 1: 6 hexagons
        for k in range(6):
            angle = np.radians(60 * k)
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            a1 = al1 + angle
            hexes.append(hex_vertices(x1, y1, 1.0, a1))
        
        # Orbit 2: 6 hexagons
        for k in range(6):
            angle = np.radians(60 * k)
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            a2 = al2 + angle
            hexes.append(hex_vertices(x2, y2, 1.0, a2))
        
        return hexes

    def compute_min_side_fixed(hexes, outer_angle_rad=0.0):
        """
        Calculate min side length of outer regular hexagon (centered at origin)
        with given orientation that contains all given hexagons.
        """
        normal_angles = outer_angle_rad + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, float(np.max(proj)))
        
        return max_proj * 2.0 / np.sqrt(3.0)

    def compute_min_side_smooth(hexes, temperature=0.01, outer_angle_rad=0.0):
        """Smooth version using log-sum-exp for differentiability."""
        normal_angles = outer_angle_rad + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Collect all vertex projections onto each normal
        all_projs = []
        for h in hexes:
            proj = h @ normals.T  # shape (6, 6)
            all_projs.append(proj)
        
        all_projs = np.concatenate(all_projs, axis=0)  # shape (72, 6)
        
        # For each of the 6 normals, compute smooth max of projections
        smooth_maxes = np.array([smooth_max(all_projs[:, i], temperature) for i in range(6)])
        
        # The side length is determined by the maximum of the 6 smooth maxes
        max_smooth = smooth_max(smooth_maxes, temperature)
        
        return max_smooth * 2.0 / np.sqrt(3.0)

    def compute_objective(params, temperature=0.01, penalty_weight=50.0, outer_angle_rad=0.0):
        """
        Objective: min_side + penalty for overlaps.
        params: [r1, theta1, alpha1, r2, theta2, alpha2]
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        # Smooth min side length
        min_s = compute_min_side_smooth(hexes, temperature, outer_angle_rad)
        
        # Overlap penalty
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
                    # Smooth penalty using softplus
                    penalty += penalty_weight * (np.log1p(np.exp(overlap * 10)) / 10)
                    
        return min_s + penalty

    def compute_objective_exact(params, outer_angle_rad=0.0):
        """Exact objective for final evaluation."""
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        min_s = compute_min_side_fixed(hexes, outer_angle_rad)
        
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
                    
        return min_s + penalty

    def optimize_outer_rotation(hexes):
        """
        Optimize the outer hexagon rotation angle to minimize side length.
        Uses golden section search over [0, 60) degrees.
        """
        best_angle = 0.0
        best_s = np.inf
        
        gr = (np.sqrt(5) - 1) / 2
        a, b = 0.0, 60.0
        
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = compute_min_side_fixed(hexes, np.radians(x1))
        f2 = compute_min_side_fixed(hexes, np.radians(x2))
        
        for _ in range(40):
            if time.time() > deadline - 5:
                break
                
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = compute_min_side_fixed(hexes, np.radians(x1))
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = compute_min_side_fixed(hexes, np.radians(x2))
        
        # Check additional candidate angles
        candidates = [0.0, 15.0, 30.0, 45.0, (a + b) / 2]
        for ang in candidates:
            s = compute_min_side_fixed(hexes, np.radians(ang))
            if s < best_s:
                best_s = s
                best_angle = ang
        
        return best_s, best_angle

    # --- Optimization ---
    
    bounds = [
        (0.5, 5.0),   # r1
        (0.0, 2*np.pi), # theta1
        (0.0, np.pi/3), # alpha1
        (0.5, 5.0),   # r2
        (0.0, 2*np.pi), # theta2
        (0.0, np.pi/3), # alpha2
    ]
    
    best_s = np.inf
    best_params = None
    best_outer_angle_deg = 0.0
    
    # Phase 1: Coarse global search with smooth objective (wider temperature)
    if time.time() < deadline - 80:
        try:
            res = differential_evolution(
                lambda p: compute_objective(p, temperature=0.05, penalty_weight=20.0),
                bounds,
                maxiter=150,
                popsize=30,
                tol=1e-4,
                seed=42,
                polish=False
            )
            if res.fun < best_s:
                best_s = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Medium refinement with tighter smooth objective
    if best_params is not None and time.time() < deadline - 50:
        try:
            res2 = differential_evolution(
                lambda p: compute_objective(p, temperature=0.01, penalty_weight=50.0),
                bounds,
                x0=best_params,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                seed=43,
                polish=False
            )
            if res2.fun < best_s:
                best_s = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Local polish with L-BFGS-B
    if best_params is not None and time.time() < deadline - 30:
        try:
            res_local = minimize(
                lambda p: compute_objective(p, temperature=0.005, penalty_weight=100.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
    
    # Phase 4: Optimize outer rotation and do final exact evaluation
    if best_params is not None and time.time() < deadline - 15:
        hexes = get_hexes_from_params(best_params)
        outer_s, outer_angle_deg = optimize_outer_rotation(hexes)
        
        # Now try to improve inner positions with the fixed outer angle
        outer_angle_rad = np.radians(outer_angle_deg)
        
        try:
            res_rot = minimize(
                lambda p: compute_objective(p, temperature=0.002, penalty_weight=200.0, outer_angle_rad=outer_angle_rad),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if res_rot.fun < best_s:
                best_params = res_rot.x.copy()
                hexes = get_hexes_from_params(best_params)
                outer_s, outer_angle_deg = optimize_outer_rotation(hexes)
        except Exception:
            pass
        
        # Final exact evaluation
        if best_params is not None:
            hexes = get_hexes_from_params(best_params)
            final_s, final_angle = optimize_outer_rotation(hexes)
            
            # Check validity with exact overlap check
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    min1 = hexes[i].min(axis=0)
                    max1 = hexes[i].max(axis=0)
                    min2 = hexes[j].min(axis=0)
                    max2 = hexes[j].max(axis=0)
                    if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                        continue
                    overlap = sat_overlap(hexes[i], hexes[j])
                    if overlap > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_s = final_s
                best_outer_angle_deg = final_angle
    
    # Fallback if no good solution found
    if best_params is None:
        # Try several random starts with known good configurations
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            p = np.array([
                np.random.uniform(1.5, 3.0), np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi/6),
                np.random.uniform(2.5, 4.5), np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi/6)
            ])
            val = compute_objective_exact(p)
            if val < best_s:
                best_s = val
                best_params = p.copy()
                
                try:
                    res_local = minimize(
                        lambda q: compute_objective(q, temperature=0.01, penalty_weight=50.0),
                        p,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200}
                    )
                    if res_local.fun < best_s:
                        best_s = res_local.fun
                        best_params = res_local.x.copy()
                except Exception:
                    pass
        
        if best_params is not None:
            hexes = get_hexes_from_params(best_params)
            outer_s, outer_angle_deg = optimize_outer_rotation(hexes)
            best_s = outer_s
            best_outer_angle_deg = outer_angle_deg
    
    # Generate the 12 inner hexagons
    r1, th1, al1, r2, th2, al2 = best_params
    
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        x1 = r1 * np.cos(th1 + angle)
        y1 = r1 * np.sin(th1 + angle)
        a1 = al1 + angle
        inner.append([x1, y1, np.degrees(a1)])
        
        x2 = r2 * np.cos(th2 + angle)
        y2 = r2 * np.sin(th2 + angle)
        a2 = al2 + angle
        inner.append([x2, y2, np.degrees(a2)])
    
    # Compute the actual minimum outer side length
    hexes = get_hexes_from_params(best_params)
    outer_side, outer_angle_deg = optimize_outer_rotation(hexes)
    
    outer_center = (0.0, 0.0)
    
    return inner, outer_center, outer_side, outer_angle_deg
