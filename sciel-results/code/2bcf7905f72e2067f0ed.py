import numpy as np
import time
from scipy.optimize import minimize

N = 12
SQRT3 = np.sqrt(3)
APOTHEM_INNER = SQRT3 / 2.0
APOTHEM_OUTER_FACTOR = SQRT3 / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_outer_normals(angle_rad):
        # Normals to the edges of the outer hexagon
        # Edge normals are at angle + k * 60 degrees
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def compute_outer_side_fast(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_rad):
        """
        Fast computation of required S.
        For each inner hex, for each vertex, project onto outer normals.
        S_req = max( dot(v - c_out, n_k) / APOTHEM_OUTER_FACTOR )
        """
        outer_normals = get_outer_normals(outer_angle_rad)
        max_proj = 0.0
        for i in range(N):
            verts = hexagon_vertices(inner_centers[i, 0], inner_centers[i, 1], 1.0, inner_angles[i])
            # Project all vertices onto all 6 normals
            # verts: 6x2, outer_normals: 6x2
            # We want max over k of dot(verts[:, j] - center, normals[k])
            # This is equivalent to max over k of (dot(verts, normals[k]) - dot(center, normals[k]))
            # Actually, simpler:
            # dist_k = max_j dot(verts[j] - center, normals[k])
            # We need S such that dist_k <= S * APOTHEM_OUTER_FACTOR
            # So S >= dist_k / APOTHEM_OUTER_FACTOR
            
            # Vectorized:
            # verts_centered = verts - [outer_cx, outer_cy]
            # projections = verts_centered @ outer_normals.T  # 6x6 matrix, proj[j, k]
            # max_proj_per_normal = np.max(projections, axis=0) # 6 values
            # max_proj = max(max_proj, np.max(max_proj_per_normal))
            
            verts_centered = verts - np.array([outer_cx, outer_cy])
            projections = verts_centered @ outer_normals.T
            max_projs = np.max(projections, axis=0)
            current_max = np.max(max_projs)
            if current_max > max_proj:
                max_proj = current_max
        
        return max_proj / APOTHEM_OUTER_FACTOR

    def sat_distance_fast(poly1, poly2):
        """
        Fast SAT distance for two convex polygons.
        Returns min separation (positive if disjoint, negative if overlapping).
        """
        max_gap = -np.inf
        
        # Get normals
        def get_normals(poly):
            normals = []
            n = len(poly)
            for i in range(n):
                v1 = poly[i]
                v2 = poly[(i + 1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normals.append(normal / norm)
            return normals
        
        normals1 = get_normals(poly1)
        normals2 = get_normals(poly2)
        
        for normal in normals1:
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap = min(min2 - max1, min1 - max2)
            if gap > max_gap:
                max_gap = gap
                
        for normal in normals2:
            proj1 = np.dot(poly1, normal)
            proj2 = np.dot(poly2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap = min(min2 - max1, min1 - max2)
            if gap > max_gap:
                max_gap = gap
                
        return max_gap

    def objective_exact(params):
        """
        Objective: Outer Side + Penalty for overlaps.
        params: [x0..x11, y0..y11, a0..a11, outer_cx, outer_cy, outer_angle, S]
        We optimize S directly? No, S is determined by the positions.
        But we can include S in params to allow the optimizer to "grow" the container if it helps?
        No, the task is to find the MINIMUM S.
        So we should minimize the computed S, subject to constraints.
        However, if we just minimize computed S, the hexagons will overlap.
        So we need a penalty for overlap.
        
        Let's define objective as:
        S_computed + Lambda * Overlap_Penalty
        
        But S_computed is not smooth (max function).
        Better: Minimize S, with constraints.
        Or minimize S + Penalty.
        
        Let's use S as a parameter?
        If we fix positions, S is fixed.
        If we vary positions, S varies.
        
        Let's stick to:
        obj = S_computed + Penalty_Overlap
        """
        if time.time() > deadline:
            return 1e6
            
        x = params[0:12]
        y = params[12:24]
        a = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        
        # Compute S
        S = compute_outer_side_fast(np.column_stack([x, y]), a, outer_cx, outer_cy, outer_a)
        
        # Compute Overlap Penalty
        penalty = 0.0
        # Pre-compute vertices for all inner hexes
        verts_list = []
        for i in range(N):
            verts_list.append(hexagon_vertices(x[i], y[i], 1.0, a[i]))
            
        for i in range(N):
            for j in range(i + 1, N):
                dist = sat_distance_fast(verts_list[i], verts_list[j])
                if dist < 0:
                    penalty += 100.0 * dist * dist
                    
        return S + penalty

    def objective_smooth(params):
        """
        Smooth objective for L-BFGS-B.
        Uses center distance for overlap proxy.
        """
        if time.time() > deadline:
            return 1e6
            
        x = params[0:12]
        y = params[12:24]
        a = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        
        S = compute_outer_side_fast(np.column_stack([x, y]), a, outer_cx, outer_cy, outer_a)
        
        penalty = 0.0
        # Proxy overlap: min center distance approx 1.7
        # If d < 1.7, penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1.7:
                    penalty += 100.0 * (1.7 - d) * (1.7 - d)
                    
        return S + penalty

    # Initial configurations
    configs = []
    
    # Config 1: 3x4 Grid
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * SQRT3
            y = r * 1.5
            if r % 2 == 1:
                x += SQRT3 / 2
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Circular
    cx, cy, ca = [], [], []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 3.0
        cx.append(r * np.cos(angle))
        cy.append(r * np.sin(angle))
        ca.append(angle)
    cx = np.array(cx); cy = np.array(cy); ca = np.array(ca)
    cx -= np.mean(cx); cy -= np.mean(cy)
    configs.append(np.concatenate([cx, cy, ca, [0.0, 0.0, 0.0]]))

    # Config 3: Random
    np.random.seed(42)
    rx = np.random.uniform(-4, 4, 12)
    ry = np.random.uniform(-4, 4, 12)
    ra = np.random.uniform(-np.pi, np.pi, 12)
    configs.append(np.concatenate([rx, ry, ra, [0.0, 0.0, 0.0]]))

    best_params = None
    best_score = 1e9

    # Phase 1: L-BFGS-B on Smooth Objective
    for cfg in configs:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                objective_smooth, cfg, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass

    # Phase 2: Refine with Exact Objective using Nelder-Mead
    if best_params is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                objective_exact, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract best solution
    if best_params is None:
        # Fallback
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0

    x = best_params[0:12]
    y = best_params[12:24]
    a = best_params[24:36]
    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_a = best_params[38]
    
    inner = [[x[i], y[i], np.degrees(a[i])] for i in range(N)]
    
    # Final Validation
    verts_list = [hexagon_vertices(x[i], y[i], 1.0, a[i]) for i in range(N)]
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            dist = sat_distance_fast(verts_list[i], verts_list[j])
            if dist < -1e-6:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # If invalid, try to fix by scaling up or using fallback
        # For now, return the best found even if slightly invalid?
        # No, better to return a valid loose packing.
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0

    S = compute_outer_side_fast(np.column_stack([x, y]), a, outer_cx, outer_cy, outer_a)
    
    return inner, (outer_cx, outer_cy), S, np.degrees(outer_a)
