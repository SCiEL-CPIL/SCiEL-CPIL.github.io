import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
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

    # Generate candidate configurations
    candidates = []
    # Standard hexagonal cluster (7-hex ring + center + 3 more)
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(2,1),(-2,1),(0,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,2)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(1,2),(2,2),(0,1),(-1,1),(0,3),(3,1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(0,3),(3,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    # Cross
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # 4x3 minus corners
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # Staircase
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Compact 3x3+2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Star
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Asymmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # Compact with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # Wide base
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # Tall narrow
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    # Star with asymmetric outer
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    # Star with diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    # Dense cluster
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    # Zigzag
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # Star with top cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # Trapezoid
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # Wide top
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    # Star with inner diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    # 4x3 variant
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Wide with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Symmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Very tall
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    # 3x3+row
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Dense 4-wide
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # Star with asymmetric bottom
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    # Tall with top row
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    # Additional
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    # Compact diamond
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(0,-2),(2,-1)])
    # Hex ring + center + 4
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Compact L-shape
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,-1),(2,1)])
    # Dense 3x4
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Asymmetric compact
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(-1,1),(3,1),(0,-1)])
    # Tight cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,1)])
    # Offset rows
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Compact V
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(0,1),(-1,0)])
    # Dense asymmetric
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Ultra compact
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,2)])
    # Additional compact shapes
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(2,3),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(1,3),(2,3),(0,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(0,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(3,0)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,1),(2,-1),(-2,1)])
    # 4+3+4+2
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # 2+3+2+3+3
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(2,3),(3,3),(2,2)])
    # More diverse compact shapes
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(0,3),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])

    def calc_S_for_beta(pts, beta):
        """LP for minimum enclosing hexagon with given rotation."""
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

    def fast_min_enclosing_hex(pts):
        """Fast iterative method for minimum enclosing hexagon."""
        if len(pts) < 2:
            cx = float(pts[0, 0]) if len(pts) > 0 else 0.0
            cy = float(pts[0, 1]) if len(pts) > 0 else 0.0
            return (cx, cy), 0.0

        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        best_cx, best_cy, best_D = cx, cy, float('inf')

        for beta in [0.0, math.pi/12, math.pi/6]:
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])

            for it in range(200):
                pts_centered = pts - np.array([cx, cy])
                projs = pts_centered @ normals.T
                abs_projs = np.abs(projs)
                D = np.max(abs_projs)

                if D < best_D - 1e-15:
                    best_D = D
                    best_cx, best_cy = cx, cy

                active_normals = []
                for k in range(6):
                    if np.max(projs[:, k]) >= D - 1e-12:
                        active_normals.append(normals[k])
                    if np.min(projs[:, k]) <= -D + 1e-12:
                        active_normals.append(-normals[k])

                if not active_normals:
                    break

                active_arr = np.array(active_normals)
                subgrad_x = np.mean(active_arr[:, 0])
                subgrad_y = np.mean(active_arr[:, 1])

                step = min(0.1, D / (10 * (abs(subgrad_x) + abs(subgrad_y) + 1e-10)))
                new_cx = cx + step * subgrad_x
                new_cy = cy + step * subgrad_y

                new_pts_centered = pts - np.array([new_cx, new_cy])
                new_projs = new_pts_centered @ normals.T
                new_D = np.max(np.abs(new_projs))

                if new_D < D - 1e-14:
                    cx, cy = new_cx, new_cy
                else:
                    found = False
                    for factor in [0.5, 0.25, 0.1]:
                        s = step * factor
                        tcx, tcy = cx + s * subgrad_x, cy + s * subgrad_y
                        tpc = pts - np.array([tcx, tcy])
                        tp = tpc @ normals.T
                        tD = np.max(np.abs(tp))
                        if tD < D - 1e-14:
                            cx, cy = tcx, tcy
                            found = True
                            break
                    if not found:
                        break

        return (best_cx, best_cy), best_D

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

    def hexagon_vertices(cx, cy, angle_deg):
        """Return 6 vertices of unit hexagon centered at (cx,cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            ang = a + k * math.pi / 3.0
            verts.append((cx + math.cos(ang), cy + math.sin(ang)))
        return verts

    def poly_overlap(v1, v2, tol=1e-7):
        """Check if two convex polygons overlap (interior intersection)."""
        xs1 = [v[0] for v in v1]; ys1 = [v[1] for v in v1]
        xs2 = [v[0] for v in v2]; ys2 = [v[1] for v in v2]
        if max(xs1) < min(xs2) + tol or max(xs2) < min(xs1) + tol:
            return False
        if max(ys1) < min(ys2) + tol or max(ys2) < min(ys1) + tol:
            return False
        for poly in (v1, v2):
            n = len(poly)
            for i in range(n):
                ax1, ay1 = poly[i]
                ax2, ay2 = poly[(i+1) % n]
                dx = ax2 - ax1
                dy = ay2 - ay1
                nx, ny = -dy, dx
                norm = math.sqrt(nx*nx + ny*ny)
                if norm < 1e-12:
                    continue
                nx /= norm; ny /= norm
                proj1 = [v[0]*nx + v[1]*ny for v in v1]
                proj2 = [v[0]*nx + v[1]*ny for v in v2]
                if max(proj1) < min(proj2) + tol or max(proj2) < min(proj1) + tol:
                    return False
        return True

    def check_all_disjoint(pts, angles_deg, tol=1e-7):
        """Check all pairs of hexagons are disjoint."""
        n = len(pts)
        verts_list = []
        for i in range(n):
            verts_list.append(hexagon_vertices(pts[i,0], pts[i,1], angles_deg[i]))
        for i in range(n):
            for j in range(i+1, n):
                if poly_overlap(verts_list[i], verts_list[j], tol):
                    return False
        return True

    def continuous_relax(pts, beta, center, max_iters=150):
        """Try to improve by moving hexagons continuously (off-lattice)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        n_pts = len(pts)
        cur_pts = pts.copy()
        cur_beta = beta
        cur_center = center
        cur_S = exact_S(cur_pts, cur_center[0], cur_center[1], cur_beta)
        
        angles_deg = [0.0] * n_pts
        
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            for i in range(n_pts):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                for dx, dy in [(0.01,0),(-0.01,0),(0,0.01),(0,-0.01),
                               (0.007,0.007),(-0.007,0.007),(0.007,-0.007),(-0.007,-0.007),
                               (0.003,0),(-0.003,0),(0,0.003),(0,-0.003),
                               (0.005,0),(-0.005,0),(0,0.005),(0,-0.005)]:
                    test_pts = cur_pts.copy()
                    test_pts[i, 0] += dx
                    test_pts[i, 1] += dy
                    
                    if not check_all_disjoint(test_pts, angles_deg):
                        continue
                    
                    S_test, c_test, _ = calc_S_for_beta(test_pts, cur_beta)
                    if S_test < cur_S - 1e-10:
                        try:
                            res = minimize_scalar(lambda b: calc_S_for_beta(test_pts, b)[0],
                                                 bounds=(0, math.pi/6), method='bounded',
                                                 options={'maxiter': 200, 'xatol': 1e-10})
                            if res.fun < S_test:
                                S_test = res.fun
                                cur_beta = res.x
                                _, c_test, _ = calc_S_for_beta(test_pts, cur_beta)
                        except Exception:
                            pass
                        
                        cur_pts = test_pts
                        cur_S = S_test
                        cur_center = c_test
                        improved = True
                        break
        
        if cur_S < exact_S(pts, center[0], center[1], beta) - 1e-10:
            return cur_pts, cur_beta, cur_center, cur_S
        return None

    def newton_polish_center(pts, beta):
        """High-precision 2D Newton polish for the center at fixed optimal rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        S_init, center_init, _ = calc_S_for_beta(pts, beta)
        if center_init is None:
            return None, None
        
        cx, cy = center_init
        cur_S = exact_S(pts, cx, cy, beta)
        
        for iteration in range(200):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T
            abs_projs = np.abs(projs)
            D = np.max(abs_projs)
            
            active_normals = []
            for k in range(6):
                if np.max(projs[:, k]) >= D - 1e-13:
                    active_normals.append(normals[k])
                if np.min(projs[:, k]) <= -D + 1e-13:
                    active_normals.append(-normals[k])
            
            if not active_normals:
                break
            
            active_arr = np.array(active_normals)
            subgrad_x = -np.mean(active_arr[:, 0])
            subgrad_y = -np.mean(active_arr[:, 1])
            
            best_step = None
            best_S = cur_S
            
            for trial in range(20):
                step = 0.3 / (trial + 1)
                new_cx = cx - step * subgrad_x
                new_cy = cy - step * subgrad_y
                S_trial = exact_S(pts, new_cx, new_cy, beta)
                if S_trial < best_S:
                    best_S = S_trial
                    best_step = (new_cx, new_cy)
            
            if best_step is None:
                break
            
            old_cx, old_cy = cx, cy
            cx, cy = best_step
            cur_S = best_S
            
            if abs(cx - old_cx) < 1e-16 and abs(cy - old_cy) < 1e-16:
                break
        
        return (cx, cy), cur_S

    def optimize_shape(pts):
        """Multi-stage optimization for a fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Stage 0: Fast initial estimate
        fast_center, fast_D = fast_min_enclosing_hex(pts)
        if fast_D < float('inf'):
            fast_S = (fast_D + APOTHEM) / APOTHEM
            best_S = min(best_S, fast_S)
            if best_center is None or fast_S < best_S:
                best_center = fast_center

        # Stage 1: Finer grid search over beta in [0, pi/6]
        n_grid = 121
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        # Stage 2: Scalar refinement of beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 500, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Stage 3: Joint (cx, cy, beta) optimization with smooth surrogate
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                temperatures = [0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
                for temp in temperatures:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break

                    def smooth_obj(params, t=temp):
                        cx, cy, beta = params
                        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        return smooth_calc_S(pts, cx, cy, beta, temperature=t)

                    start_betas = [best_beta, best_beta + 0.03, best_beta - 0.03, 0.0, math.pi / 12, best_beta + 0.06, best_beta - 0.06]
                    for sb in start_betas:
                        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                            break
                        init = [best_center[0], best_center[1], sb]
                        try:
                            res = minimize(smooth_obj, init, method='Nelder-Mead',
                                         options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
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

        # Stage 3.5: Continuous relaxation - move hexagons off lattice
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                relax_result = continuous_relax(pts, best_beta, best_center, max_iters=100)
                if relax_result is not None:
                    rp, rb, rc, rS = relax_result
                    if rS < best_S:
                        best_S = rS
                        best_beta = rb
                        best_center = rc
                        pts = rp
        except Exception:
            pass

        # Stage 4: High-precision polish with exact objective
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def exact_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    return exact_S(pts, cx, cy, beta)

                delta = 0.005
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + delta],
                    [best_center[0], best_center[1], best_beta - delta],
                    [best_center[0] + delta, best_center[1], best_beta],
                    [best_center[0] - delta, best_center[1], best_beta],
                    [best_center[0], best_center[1] + delta, best_beta],
                    [best_center[0], best_center[1] - delta, best_beta],
                    [best_center[0] + delta*0.5, best_center[1] + delta*0.5, best_beta],
                    [best_center[0] - delta*0.5, best_center[1] - delta*0.5, best_beta],
                    [best_center[0] + delta*0.5, best_center[1] - delta*0.5, best_beta],
                    [best_center[0] - delta*0.5, best_center[1] + delta*0.5, best_beta],
                    [best_center[0], best_center[1], best_beta + delta*0.5],
                    [best_center[0], best_center[1], best_beta - delta*0.5],
                ]

                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    try:
                        res = minimize(exact_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 12000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                        if res.fun < best_S:
                            best_S = res.fun
                            best_center = (res.x[0], res.x[1])
                            best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    except Exception:
                        pass
        except Exception:
            pass

        # Stage 4.5: Newton polish for center at fixed beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                center_new, S_new = newton_polish_center(pts, best_beta)
                if center_new is not None and S_new is not None and S_new < best_S:
                    best_S = S_new
                    best_center = center_new
        except Exception:
            pass

        # Stage 5: Final LP-based center refinement
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                S_final, center_final, D_final = calc_S_for_beta(pts, best_beta)
                if S_final < best_S:
                    best_S = S_final
                    best_center = center_final
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    def quick_bbox_score(axial_list):
        pts = axial_to_cartesian(axial_list)
        return np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1])

    sorted_candidates = sorted(candidates, key=quick_bbox_score)

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for c in sorted_candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Evaluate candidates
    for idx, axial in enumerate(unique_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
        
        if idx >= 15 and best_global_S < 3.5:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 10:
                break

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(unique_candidates[0])
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

    # Final: compute side length with tiny safety scale
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

    # Verify disjointness of final configuration
    angles_deg_final = [angle_deg] * len(pts_scaled)
    if not check_all_disjoint(pts_scaled, angles_deg_final, tol=1e-6):
        pts_final = best_global_pts.copy()
        max_proj_orig = 0
        for k in range(6):
            ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts_final[:, 0] - cx) * nx + (pts_final[:, 1] - cy) * ny
            max_proj_orig = max(max_proj_orig, np.max(projs), -np.min(projs))
        S_scaled = (max_proj_orig + APOTHEM) / APOTHEM
    else:
        pts_final = pts_scaled

    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
