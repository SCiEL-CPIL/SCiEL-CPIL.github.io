import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0
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

    # Diverse compact configurations of 11 unit hexagons
    candidates = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        # Additional compact configurations
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(2,1),(-2,1),(-1,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,1)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,2),(-1,1),(3,1)],
        [(0,0),(1,0),(0,1),(1,1),(-1,1),(0,2),(1,2),(-1,2),(2,1),(-2,1),(0,-1)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,1),(-2,1),(1,-1),(-1,-1)],
    ]

    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        return (max_proj + APOTHEM) / APOTHEM

    estimated_scores = [(quick_estimate(ax), ax) for ax in candidates]
    estimated_scores.sort(key=lambda x: x[0])

    def calc_S_for_beta(pts, beta):
        """LP: minimum enclosing hexagon (center + apothem) at rotation beta."""
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
                D = float(res.x[2])
                return (D + APOTHEM) / APOTHEM, (float(res.x[0]), float(res.x[1])), D
            return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = (pts - np.array([cx, cy])) @ normals.T
        D = float(np.max(np.abs(projs)))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth log-sum-exp surrogate for the nonsmooth max |proj| objective."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = (pts - np.array([cx, cy])) @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = float(np.max(all_projs))
        if max_val > 200:
            return float('inf')
        shifted = np.clip((all_projs - max_val) / temperature, -50, 50)
        D = temperature * (np.log(np.sum(np.exp(shifted))) + max_val / temperature)
        return (D + APOTHEM) / APOTHEM

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        # Coarse grid over beta (period pi/6)
        for i in range(73):
            beta = i * (math.pi / 6) / 72
            S, center, _ = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S, best_beta, best_center = S, beta, center

        # Finer grid around the coarse best
        for i in range(25):
            beta = best_beta - math.pi/72 + i * (2 * math.pi / 72) / 24
            if beta < 0 or beta > math.pi/6:
                continue
            S, center, _ = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S, best_beta, best_center = S, beta, center

        # Bounded scalar refinement
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                lo = max(0.0, best_beta - math.pi/72)
                hi = min(math.pi/6, best_beta + math.pi/72)
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0],
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 500, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_beta = float(res.x)
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass

        # Multi-method polish (exact objective)
        def objective(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            D = float(np.max(np.abs((pts - np.array([cx, cy])) @ normals.T)))
            return (D + APOTHEM) / APOTHEM

        polish_starts = [
            [best_center[0], best_center[1], best_beta],
            [best_center[0], best_center[1], best_beta + 0.01],
            [best_center[0], best_center[1], best_beta - 0.01],
            [best_center[0] + 0.01, best_center[1], best_beta],
            [best_center[0], best_center[1] + 0.01, best_beta],
            [best_center[0] - 0.01, best_center[1], best_beta],
            [best_center[0], best_center[1] - 0.01, best_beta],
            [best_center[0] + 0.005, best_center[1] + 0.005, best_beta],
            [best_center[0] - 0.005, best_center[1] - 0.005, best_beta],
            [best_center[0] + 0.002, best_center[1] + 0.002, best_beta + 0.002],
            [best_center[0] - 0.002, best_center[1] - 0.002, best_beta - 0.002],
        ]

        for init in polish_starts:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                res = minimize(objective, init, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-15})
                if res.fun < best_S:
                    best_S = float(res.fun)
                    best_center = (float(res.x[0]), float(res.x[1]))
                    best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                continue
            try:
                res2 = minimize(objective, [best_center[0], best_center[1], best_beta],
                                method='SLSQP', options={'maxiter': 5000, 'ftol': 1e-16})
                if res2.fun < best_S:
                    best_S = float(res2.fun)
                    best_center = (float(res2.x[0]), float(res2.x[1]))
                    best_beta = ((float(res2.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                continue
            try:
                res3 = minimize(objective, [best_center[0], best_center[1], best_beta],
                                method='COBYLA', options={'maxiter': 5000, 'rhobeg': 1e-8})
                if res3.fun < best_S:
                    best_S = float(res3.fun)
                    best_center = (float(res3.x[0]), float(res3.x[1]))
                    best_beta = ((float(res3.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass

        # Smooth-surrogate polish (helps escape nonsmooth plateaus)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def smooth_obj(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return smooth_calc_S(pts, cx, cy, beta, temperature=0.03)
                for init in polish_starts[:6]:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(smooth_obj, init, method='Nelder-Mead',
                                   options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-12})
                    S_ex = objective([float(res.x[0]), float(res.x[1]), float(res.x[2])])
                    if S_ex < best_S:
                        best_S = S_ex
                        best_center = (float(res.x[0]), float(res.x[1]))
                        best_beta = ((float(res.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    PER_CANDIDATE_BUDGET = 1.6
    for _, axial in estimated_scores:
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            break
        remaining = TIME_LIMIT - SAFETY_MARGIN - elapsed
        if remaining < PER_CANDIDATE_BUDGET and best_global_S != float('inf'):
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Tiny scale-up so the reported side length strictly contains the hexagons
    SCALE = 1.0 + 1e-12
    pts_scaled = np.array([(cx + (x - cx) * SCALE, cy + (y - cy) * SCALE)
                           for x, y in best_global_pts])

    max_proj = 0.0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    side = (max_proj + APOTHEM) / APOTHEM

    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts_scaled]
    center = [float(cx), float(cy)]

    return inner, center, float(side), float(angle_deg)
