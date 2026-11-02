import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0
EPS = 1e-9

def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # ---- fixed lattice candidates (axial coordinates) as warm starts / fallbacks ----
    candidates = []
    axial_1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
    candidates.append(axial_1)
    axial_2 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_2)
    axial_3 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)]
    candidates.append(axial_3)
    axial_4 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)]
    candidates.append(axial_4)
    axial_5 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_5)
    axial_6 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)]
    candidates.append(axial_6)
    axial_7 = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_7)
    axial_8 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
    candidates.append(axial_8)
    axial_9 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)]
    candidates.append(axial_9)
    axial_10 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)]
    candidates.append(axial_10)
    axial_11 = [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)]
    candidates.append(axial_11)
    axial_12 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_12)
    axial_13 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)]
    candidates.append(axial_13)
    axial_14 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)]
    candidates.append(axial_14)
    axial_15 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)]
    candidates.append(axial_15)
    axial_16 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)]
    candidates.append(axial_16)
    axial_17 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)]
    candidates.append(axial_17)
    axial_18 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)]
    candidates.append(axial_18)
    axial_19 = [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)]
    candidates.append(axial_19)
    axial_20 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)]
    candidates.append(axial_20)
    axial_21 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]
    candidates.append(axial_21)
    axial_22 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)]
    candidates.append(axial_22)
    axial_23 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
    candidates.append(axial_23)
    axial_24 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_24)
    axial_25 = [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)]
    candidates.append(axial_25)
    axial_26 = [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)]
    candidates.append(axial_26)
    axial_27 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)]
    candidates.append(axial_27)
    axial_28 = [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)]
    candidates.append(axial_28)
    axial_29 = [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(0,4)]
    candidates.append(axial_29)
    axial_30 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1),(0,2)]
    candidates.append(axial_30)
    axial_31 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4)]
    candidates.append(axial_31)
    axial_32 = [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4),(1,4)]
    candidates.append(axial_32)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # ---- support function of a unit hexagon (flat-top) with rotation phi ----
    def support(phi):
        # vertices at angles phi + k*pi/3, k=0..5  -> 6 support values (max over vertices)
        th = phi + np.arange(6) * (math.pi / 3.0)
        return np.cos(th)

    # ---- exact outer-half-apothem D for fixed positions+angles (min over outer center & rotation) ----
    def exact_D(pts, angs):
        best = float('inf')
        for i in range(37):
            beta = i * (math.pi / 6.0) / 36.0
            normals = np.array([
                (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)) for k in range(6)])
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub, b_ub = [], []
            for nx, ny in normals:
                for j in range(N):
                    s = support(angs[j])
                    px = pts[j, 0]; py = pts[j, 1]
                    for a in s:
                        A_ub.append([-nx, -ny, -1.0]); b_ub.append(-(px*nx + py*ny) + a)
                        A_ub.append([nx, ny, -1.0]);  b_ub.append((px*nx + py*ny) - a)
            A_ub = np.array(A_ub); b_ub = np.array(b_ub)
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                              bounds=[(None,None),(None,None),(0,None)], method='highs')
                if res.status == 0:
                    best = min(best, res.x[2])
            except Exception:
                pass
        return best

    # ---- positional optimization for a fixed outer rotation beta (convex fit + distance penalty) ----
    def optimize_positions(pts0, angs, beta):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)) for k in range(6)])
        S = np.stack([support(a) for a in angs], axis=1)   # (N,6)
        M = []
        for j in range(N):
            for nx, ny in normals:
                row = np.zeros(2*N)
                row[2*j] = nx; row[2*j+1] = ny
                M.append(row)
        M = np.array(M)
        b = np.array([-(S[j, k] + APOTHEM) for j in range(N) for k in range(6)])  # M x <= b

        def objective(v):
            p = v.reshape(N, 2)
            fit = np.max(np.maximum(M @ v - b, 0.0))
            # pairwise distance penalty (pushes apart)
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = p[i,0]-p[j,0]; dy = p[i,1]-p[j,1]
                    d = math.sqrt(dx*dx + dy*dy)
                    pen += max(0.0, 2.0 - d) ** 2
            return fit + 50.0 * pen

        best = None; best_val = float('inf')
        starts = [pts0.ravel()]
        # a few perturbed starts
        rng = np.random.RandomState(7)
        for _ in range(4):
            starts.append(pts0.ravel() + rng.normal(0.0, 0.6, size=2*N))
        for s in starts:
            if not _time_ok():
                break
            try:
                res = minimize(objective, s, method='L-BFGS-B',
                               options={'maxiter': 600, 'maxfun': 12000})
                if res.fun < best_val:
                    best_val = res.fun
                    best = res.x
            except Exception:
                pass
        if best is None:
            return pts0, angs, float('inf')
        return best.reshape(N, 2), angs, best_val

    # ---- per-configuration: optimize positions for several beta, then refine outer hex ----
    def refine_outer(pts, angs, beta0, cx0, cy0):
        def exact_obj(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6.0)) + math.pi/6.0) % (math.pi/6.0)
            normals = np.array([
                (math.cos(beta + math.pi/6.0 + k*math.pi/3.0),
                 math.sin(beta + math.pi/6.0 + k*math.pi/3.0)) for k in range(6)])
            m = 0.0
            for nx, ny in normals:
                for j in range(N):
                    s = support(angs[j])
                    proj = (pts[j,0]-cx)*nx + (pts[j,1]-cy)*ny
                    m = max(m, proj + np.max(s), -(proj + np.min(s)))
            return m
        best_D = float('inf'); best_b = beta0; best_c = (cx0, cy0)
        # coarse
        for i in range(37):
            b = i*(math.pi/6.0)/36.0
            # center at centroid for coarse
            cx = float(np.mean(pts[:,0])); cy = float(np.mean(pts[:,1]))
            D = exact_obj([cx, cy, b])
            if D < best_D:
                best_D, best_b, best_c = D, b, (cx, cy)
        # refine center & beta
        try:
            if _time_ok():
                res = minimize(exact_obj, [best_c[0], best_c[1], best_b], method='Nelder-Mead',
                               options={'maxiter': 4000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_D:
                    best_D = res.fun
                    best_c = (res.x[0], res.x[1])
                    best_b = ((res.x[2] % (math.pi/6.0)) + math.pi/6.0) % (math.pi/6.0)
        except Exception:
            pass
        return best_D, best_b, best_c

    # ================= main search =================
    best_D = float('inf'); best_pts = None; best_angs = None
    best_beta = 0.0; best_center = (0.0, 0.0)

    for axial in candidates:
        if not _time_ok():
            break
        pts = axial_to_cartesian(axial)
        angs = np.zeros(N)
        cx0 = float(np.mean(pts[:,0])); cy0 = float(np.mean(pts[:,1]))

        # try a handful of outer rotations; optimize positions for each
        for beta in [0.0, math.pi/12.0, math.pi/6.0, math.pi/9.0, math.pi/18.0]:
            if not _time_ok():
                break
            p_opt, a_opt, _ = optimize_positions(pts, angs, beta)
            D, b, c = refine_outer(p_opt, a_opt, beta, cx0, cy0)
            if D < best_D:
                best_D = D; best_pts = p_opt; best_angs = a_opt
                best_beta = b; best_center = c

    # fallback: best lattice with exact outer hex
    if best_pts is None or not math.isfinite(best_D):
        pts = axial_to_cartesian(candidates[0]); angs = np.zeros(N)
        best_D, best_beta, best_center = refine_outer(pts, angs, 0.0,
                                                      float(np.mean(pts[:,0])),
                                                      float(np.mean(pts[:,1])))
        best_pts = pts; best_angs = angs

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    # recompute exact D with final center/beta for the side length
    normals = np.array([
        (math.cos(best_beta + math.pi/6.0 + k*math.pi/3.0),
         math.sin(best_beta + math.pi/6.0 + k*math.pi/3.0)) for k in range(6)])
    m = 0.0
    for nx, ny in normals:
        for j in range(N):
            s = support(best_angs[j])
            proj = (best_pts[j,0]-cx)*nx + (best_pts[j,1]-cy)*ny
            m = max(m, proj + np.max(s), -(proj + np.min(s)))
    D = m
    side = (D + APOTHEM) / APOTHEM

    inner = []
    for j in range(N):
        inner.append([float(best_pts[j,0]), float(best_pts[j,1]),
                      float(math.degrees(best_angs[j]))])
    center = [float(cx), float(cy)]
    return inner, center, float(side), float(angle_deg)
