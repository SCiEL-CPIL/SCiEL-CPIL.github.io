import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0
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

    def hexagon_vertices(cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        verts = []
        for k in range(6):
            angle = rot + k * math.pi / 3.0
            x = cx + side * math.cos(angle)
            y = cy + side * math.sin(angle)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, rotation_deg):
        rot = math.radians(rotation_deg)
        dx = px - cx
        dy = py - cy
        dx_local = dx * math.cos(rot) + dy * math.sin(rot)
        dy_local = -dx * math.sin(rot) + dy * math.cos(rot)
        apothem = side * APOTHEM
        for k in range(6):
            angle = k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            dot = dx_local * nx + dy_local * ny
            if dot > apothem + 1e-10:
                return False
        return True

    def hexagons_disjoint(c1, c2, s1, s2, rot1_deg, rot2_deg):
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return dist >= (s1 + s2) - 1e-9

    # Generate candidate configurations (axial coordinates for aligned hexagons)
    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
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
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(0,1),(0,-1),(1,1)])
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(1,1),(-1,1),(0,1)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(-1,1),(0,1),(1,1),(2,1),(3,1),(4,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3)])
    candidates.append([(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(2,-2)])

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation (vectorized)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

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
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def compute_convex_hull_2d(points):
        """Compute convex hull of 2D points using Andrew's monotone chain."""
        pts = sorted(set(map(tuple, points)))
        if len(pts) <= 2:
            return np.array(pts)
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        hull = lower[:-1] + upper[:-1]
        return np.array(hull)

    def rotating_calipers_optimize(pts):
        """
        Rotating calipers: compute convex hull, find all critical beta values
        where the extreme point changes, then evaluate S at each critical point.
        For a convex polygon, the support function in direction u(beta) is piecewise
        linear in beta, with breakpoints when u(beta) is perpendicular to an edge.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        hull = compute_convex_hull_2d(pts)
        n_hull = len(hull)
        
        if n_hull < 3:
            # Degenerate case
            S, center, D = calc_S_for_beta(pts, 0.0)
            return S, 0.0, center

        # The 6 normal directions for the outer hexagon at orientation beta
        # are at angles: beta + pi/6 + k*pi/3, k=0..5
        # For each hull vertex v_i, it is the maximizer of dot(v, n) when n
        # lies in the normal cone of v_i, i.e., between the outward normals
        # of the two adjacent edges.
        
        # For a convex polygon, the support function h(n) = max_i dot(v_i, n)
        # is a convex piecewise-linear function of n. When parameterizing
        # n = (cos(theta), sin(theta)) with theta = beta + pi/6 + k*pi/3,
        # h becomes piecewise in beta with breakpoints where n is perpendicular
        # to a hull edge.
        
        # Collect all critical beta values in [0, pi/6)
        critical_betas = [0.0, math.pi / 6.0]
        
        # For each hull edge, find when the normal direction is perpendicular to it
        for i in range(n_hull):
            v1 = hull[i]
            v2 = hull[(i + 1) % n_hull]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            
            # Outward normal to edge (perpendicular, pointing outward)
            # For CCW hull, outward normal is (edge_y, -edge_x) normalized
            # Actually for CCW polygon, the outward normal of edge v1->v2 is
            # perpendicular pointing away from interior.
            # Let's compute: if hull is CCW, the left normal of edge is inward,
            # so right normal is outward: n_out = (dy, -dx) / |edge|
            # Wait, let me be careful. For CCW polygon, interior is to the left.
            # So outward normal is to the right: (dy, -dx) normalized? 
            # No: if edge direction is (dx, dy), left normal is (-dy, dx), right is (dy, -dx).
            # For CCW, interior is on the left, so outward is right: (dy, -dx).
            
            # The critical condition: the hexagon normal n_k(beta) = (cos(theta_k), sin(theta_k))
            # is perpendicular to the edge, meaning n_k is parallel to the edge normal.
            # i.e., theta_k = atan2(edge_y, edge_x) + pi/2 (mod 2*pi) for one normal
            # or theta_k = atan2(edge_y, edge_x) - pi/2 for the other.
            
            # More precisely, n_k is perpendicular to edge when:
            # cos(theta_k)*dx + sin(theta_k)*dy = 0
            # => theta_k = atan2(dy, dx) + pi/2 (mod 2*pi)
            
            edge_angle = math.atan2(edge[1], edge[0])
            # The normal to the edge is at edge_angle + pi/2 and edge_angle - pi/2
            # We need theta_k = beta + pi/6 + k*pi/3 to equal edge_angle +/- pi/2 (mod 2*pi)
            # => beta = edge_angle +/- pi/2 - pi/6 - k*pi/3 (mod 2*pi)
            
            for sign in [1, -1]:
                base = edge_angle + sign * math.pi / 2.0
                for k in range(6):
                    beta_val = base - math.pi / 6.0 - k * math.pi / 3.0
                    # Normalize to [0, 2*pi) then check if in [0, pi/6)
                    beta_val = beta_val % (2 * math.pi)
                    # Also try adding multiples of pi/6 since hexagon has 6-fold symmetry
                    # Actually beta is defined mod pi/6, so we want beta in [0, pi/6)
                    # But the critical points repeat with period pi/6 in beta
                    # due to the 6-fold symmetry of the hexagon normals.
                    # Let me reconsider: theta_k = beta + pi/6 + k*pi/3
                    # As k varies, theta_k covers 6 directions spaced by pi/3.
                    # The condition is that for SOME k, theta_k is perp to edge.
                    # theta_k = edge_angle + pi/2 (mod 2*pi)
                    # beta = edge_angle + pi/2 - pi/6 - k*pi/3 (mod 2*pi)
                    # For k=0..5, these give 6 values spaced by pi/3.
                    # Since beta is mod pi/6, and pi/3 = 2*(pi/6), we get 3 distinct
                    # values mod pi/6. Let me just collect all and normalize.
                    
                    # Normalize beta_val to [0, pi/6)
                    beta_norm = beta_val % (math.pi / 6.0)
                    if beta_norm < 1e-12:
                        beta_norm = 0.0
                    elif abs(beta_norm - math.pi / 6.0) < 1e-12:
                        beta_norm = 0.0
                    critical_betas.append(beta_norm)
            
            # Also the opposite normal direction (for the -dot constraint)
            # This is already covered by sign = -1 above.
        
        # Deduplicate and sort critical betas
        critical_betas = list(set([round(b, 12) for b in critical_betas]))
        critical_betas = [b for b in critical_betas if 0.0 <= b < math.pi / 6.0 - 1e-12]
        critical_betas.sort()
        
        # Also add some extra sample points for safety
        n_extra = 20
        for i in range(n_extra + 1):
            b = i * (math.pi / 6.0) / n_extra
            b = b % (math.pi / 6.0)
            critical_betas.append(b)
        
        critical_betas = sorted(set([round(b, 12) for b in critical_betas]))
        critical_betas = [b for b in critical_betas if 0.0 <= b < math.pi / 6.0 - 1e-12]
        
        # Evaluate S at each critical beta
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        for beta in critical_betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Also do a fine grid scan to catch any missed critical points
        n_fine = 121
        for i in range(n_fine):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6.0) / (n_fine - 1)
            if beta >= math.pi / 6.0 - 1e-12:
                beta = 0.0
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        if best_center is None:
            return float('inf'), 0.0, None
        
        # Local refinement around best_beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = math.pi / 6.0 / 50.0
                lo = max(0.0, best_beta - window)
                hi = min(math.pi / 6.0 - 1e-12, best_beta + window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Second refinement with even tighter window
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = math.pi / 6.0 / 200.0
                lo = max(0.0, best_beta - window)
                hi = min(math.pi / 6.0 - 1e-12, best_beta + window)
                
                def obj_1d(b):
                    S, _, _ = calc_S_for_beta(pts, b)
                    return S
                
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-15})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    def optimize_shape(pts):
        """Hybrid optimization for fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        S1, beta1, center1 = rotating_calipers_optimize(pts)
        if center1 is not None:
            best_S = S1
            best_beta = beta1
            best_center = center1

        if best_center is None:
            return float('inf'), 0.0, None

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.05, 0.03, 0.01, 0.005]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break

                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(pts, cx, cy, beta, temperature=t)

                    start_betas = [best_beta, best_beta + 0.01, best_beta - 0.01,
                                   best_beta + 0.03, best_beta - 0.03,
                                   best_beta + 0.05, best_beta - 0.05]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
                            if res.fun < 1e10:
                                cx_r, cy_r, beta_r = res.x
                                beta_r = ((beta_r % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                                S_exact = exact_S(pts, cx_r, cy_r, beta_r)
                                if S_exact < best_S:
                                    best_S = S_exact
                                    best_beta = beta_r
                                    best_center = (cx_r, cy_r)
                        except Exception:
                            pass
        except Exception:
            pass

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                delta_vals = [0.0, 0.0005, -0.0005, 0.001, -0.001, 0.002, -0.002,
                              0.003, -0.003, 0.005, -0.005, 0.008, -0.008]
                polish_starts = []
                for db in delta_vals:
                    for dcx in [0.0, 0.0005, -0.0005, 0.001, -0.001, 0.002, -0.002]:
                        for dcy in [0.0, 0.0005, -0.0005, 0.001, -0.001, 0.002, -0.002]:
                            polish_starts.append([
                                best_center[0] + dcx,
                                best_center[1] + dcy,
                                best_beta + db
                            ])
                seen = set()
                unique_starts = []
                for p in polish_starts:
                    key = (round(p[0], 7), round(p[1], 7), round(p[2], 7))
                    if key not in seen:
                        seen.add(key)
                        unique_starts.append(p)

                # Sort by distance from current best to prioritize nearby starts
                unique_starts.sort(key=lambda p: (p[0]-best_center[0])**2 + (p[1]-best_center[1])**2 + (p[2]-best_beta)**2)

                for init in unique_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Final high-precision polish: use SLSQP with tighter bounds
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj_slsqp(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                # Try a few final SLSQP refinements
                for db in [0.0, 0.001, -0.001]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    init = [best_center[0], best_center[1], best_beta + db]
                    try:
                        res = minimize(exact_obj_slsqp, init, method='SLSQP',
                                     options={'maxiter': 1000, 'ftol': 1e-15})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        angles = math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        pts_c = pts - np.array([cx, cy])
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    scored_candidates = []
    for axial in candidates:
        est = quick_estimate(axial)
        scored_candidates.append((est, axial))
    scored_candidates.sort(key=lambda x: x[0])

    # Process top candidates with full optimization, then fill remaining time with more
    n_top = min(25, len(scored_candidates))
    for idx in range(n_top):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        est, axial = scored_candidates[idx]
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # If time remains, try more candidates with orientation-only optimization (faster)
    for idx in range(n_top, len(scored_candidates)):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        est, axial = scored_candidates[idx]
        if est >= best_global_S * 1.001:  # Skip if clearly worse
            continue
        pts = axial_to_cartesian(axial)
        S, beta, center = rotating_calipers_optimize(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
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

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    SCALE = 1.0 + 1e-14
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
