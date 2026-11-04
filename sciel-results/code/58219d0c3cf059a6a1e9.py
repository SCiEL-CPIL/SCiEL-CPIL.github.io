import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hex_vertices(cx, cy, angle_deg):
        """Get 6 vertices of a unit regular hexagon centered at (cx,cy) with rotation."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3.0
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return np.array(verts)

    def hex_overlap_dist(p1, p2, a1, a2):
        """
        Compute the minimum distance between two unit hexagons at (p1,a1) and (p2,a2).
        If they overlap, return negative overlap depth.
        Uses separating axis theorem for hexagons.
        """
        v1 = hex_vertices(p1[0], p1[1], a1)
        v2 = hex_vertices(p2[0], p2[1], a2)
        
        # Get all edge normals from both hexagons
        normals = []
        for v in [v1, v2]:
            for i in range(6):
                edge = v[(i+1)%6] - v[i]
                n = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    n = n / norm
                    normals.append(n)
        
        # Check separation along each normal
        min_overlap = float('inf')
        for n in normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            overlap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
            if overlap < 0:
                return overlap  # Separated
            min_overlap = min(min_overlap, overlap)
        
        # If we get here, they overlap. The minimum overlap is the answer.
        # But for SMT, we need the penetration depth.
        # For simplicity, use the minimum overlap found.
        return -min_overlap if min_overlap < float('inf') else 0.0

    def min_enclosing_hexagon(pts, beta):
        """Find minimum enclosing hexagon with rotation beta. Returns (S, center, D)."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs
        
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx, cy = res.x[0], res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_S(pts, cx, cy, beta, temp=0.05):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temp
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temp
        D = temp * log_sum
        return (D + APOTHEM) / APOTHEM

    # Generate candidate configurations (axial coordinates)
    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(0,-2),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,-1),(2,1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(0,1),(-1,0)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,2)])

    def quick_bbox_score(axial_list):
        pts = axial_to_cartesian(axial_list)
        return np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1])

    sorted_candidates = sorted(candidates, key=quick_bbox_score)

    def optimize_positions_and_enclosing(pts_init, angle_deg):
        """
        Optimize the positions of N hexagons (all with same angle) to minimize
        the enclosing hexagon side length, subject to non-overlap.
        """
        n = len(pts_init)
        # Variables: x0,y0,x1,y1,...,x_{n-1},y_{n-1}, cx, cy, beta
        # Total: 2n + 3
        x0 = pts_init.copy()
        cx0 = float(np.mean(x0[:, 0]))
        cy0 = float(np.mean(x0[:, 1]))
        
        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e10
            pts = params[:2*n].reshape(n, 2)
            cx, cy, beta = params[2*n], params[2*n+1], params[2*n+2]
            
            # Check non-overlap: all pairs must be separated
            # For unit hexagons at same angle, min distance between centers is 2*APOTHEM*2 = 2*sqrt(3)
            # Actually, for two unit hexagons (side=1) at the same angle, they don't overlap
            # if the distance between centers >= 2*APOTHEM*... 
            # The inradius is APOTHEM = sqrt(3)/2. Two hexagons don't overlap if
            # center distance >= 2 * APOTHEM = sqrt(3) when aligned, but it depends on angle.
            # For safety, use a conservative check: distance >= sqrt(3) - small_epsilon
            # Actually, for two regular hexagons of side 1, the minimum center distance
            # for non-overlap is 2*APOTHEM = sqrt(3) ≈ 1.732.
            
            overlap_penalty = 0.0
            min_dist_sq = 3.0 - 1e-6  # (sqrt(3))^2 - eps
            for i in range(n):
                for j in range(i+1, n):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    d_sq = dx*dx + dy*dy
                    if d_sq < min_dist_sq:
                        # Penalty for overlap
                        deficit = min_dist_sq - d_sq
                        overlap_penalty += deficit * deficit * 100.0
            
            # Compute S
            S = exact_S(pts, cx, cy, beta)
            
            if overlap_penalty > 1e-8:
                return S + overlap_penalty
            return S
        
        # Initial parameters
        init = np.concatenate([x0.ravel(), [cx0, cy0, 0.0]])
        
        best_S = float('inf')
        best_params = init.copy()
        
        # Multi-start optimization
        n_starts = 5
        for s in range(n_starts):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            if s == 0:
                start = init.copy()
            else:
                # Perturb
                start = init + np.random.normal(0, 0.1, size=len(init))
            
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_S == float('inf'):
            return None, None, None
        
        pts_opt = best_params[:2*n].reshape(n, 2)
        cx_opt, cy_opt, beta_opt = best_params[2*n], best_params[2*n+1], best_params[2*n+2]
        beta_opt = ((beta_opt % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        # Final exact S
        S_final = exact_S(pts_opt, cx_opt, cy_opt, beta_opt)
        
        return pts_opt, (cx_opt, cy_opt), S_final

    # Strategy: For each candidate, try a few angles and optimize positions
    best_global_S = float('inf')
    best_global_pts = None
    best_global_center = None
    best_global_beta = 0.0
    best_global_angle_deg = 0.0

    # Time allocation: spend most time on top candidates
    time_per_candidate = (TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)) / max(len(sorted_candidates), 1)
    time_per_candidate = min(time_per_candidate, 3.0)  # Cap at 3 seconds per candidate

    for idx, axial in enumerate(sorted_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts_init = axial_to_cartesian(axial)
        
        # Try a few angles for the inner hexagons
        angles_to_try = [0.0, 30.0, 15.0, 45.0, 60.0]
        
        for angle_deg in angles_to_try:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            result = optimize_positions_and_enclosing(pts_init, angle_deg)
            if result is not None:
                pts_opt, center_opt, S_opt = result
                if S_opt < best_global_S:
                    best_global_S = S_opt
                    best_global_pts = pts_opt
                    best_global_center = center_opt
                    best_global_beta = ((result[0].copy() if False else 0.0))  # beta from optimization
                    # Extract beta from the optimization
                    # Re-run to get beta
                    # Actually, let's just use the LP to find optimal beta
                    best_global_angle_deg = angle_deg
    
    # If we have a good result, do a final refinement
    if best_global_pts is not None and best_global_center is not None:
        cx, cy = best_global_center
        
        # Find optimal beta via scalar optimization
        def S_of_beta(beta):
            return min_enclosing_hexagon(best_global_pts, beta)[0]
        
        try:
            res_beta = minimize_scalar(S_of_beta, bounds=(0, math.pi/6), method='bounded',
                                      options={'maxiter': 200, 'xatol': 1e-12})
            if res_beta.fun < best_global_S:
                best_global_S = res_beta.fun
                best_global_beta = res_beta.x
                _, best_global_center, _ = min_enclosing_hexagon(best_global_pts, best_global_beta)
        except Exception:
            pass
        
        # Final polish: optimize center for fixed beta
        def S_of_center(params):
            return exact_S(best_global_pts, params[0], params[1], best_global_beta)
        
        try:
            res_c = minimize(S_of_center, [cx, cy], method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
            if res_c.fun < best_global_S:
                best_global_S = res_c.fun
                best_global_center = (res_c.x[0], res_c.x[1])
        except Exception:
            pass
        
        # Final LP for exact center
        S_lp, center_lp, _ = min_enclosing_hexagon(best_global_pts, best_global_beta)
        if S_lp < best_global_S:
            best_global_S = S_lp
            best_global_center = center_lp

    # Fallback
    if best_global_pts is None:
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
        best_global_angle_deg = 0.0

    cx, cy = best_global_center
    angle_deg = best_global_angle_deg
    beta_rad = best_global_beta

    # Compute final side length
    max_proj = 0
    for k in range(6):
        ang = beta_rad + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (best_global_pts[:, 0] - cx) * nx + (best_global_pts[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_final = max_proj
    S_final = (D_final + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(math.degrees(beta_rad))

    return inner, center, side, rotation
