import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
SQRT3 = math.sqrt(3.0)
PI = math.pi
SIXTH_PI = PI / 6.0


def _time_ok(margin=SAFETY_MARGIN):
    return time.time() - START_TIME < TIME_LIMIT - margin


def _axial_to_cartesian(axial):
    pts = np.empty((len(axial), 2))
    for i, (q, r) in enumerate(axial):
        pts[i, 0] = SQRT3 * (q + 0.5 * r)
        pts[i, 1] = 1.5 * r
    return pts


def _normals(beta):
    a = beta + SIXTH_PI
    return np.column_stack([np.cos(a + np.arange(6) * (PI / 3.0)),
                            np.sin(a + np.arange(6) * (PI / 3.0))])


def _solve_lp(pts, beta):
    """Exact LP: min D s.t. for all normals n, -D <= n.(p-c) <= D.
    Returns (S, cx, cy, D). S=(D+APOTHEM)/APOTHEM."""
    if not _time_ok():
        return float('inf'), None, None, None
    nrm = _normals(beta)                       # (6,2)
    c = np.zeros(3)
    c[2] = 1.0
    A = np.empty((12 * len(pts), 3))
    b = np.empty(12 * len(pts))
    row = 0
    for k in range(6):
        nx, ny = nrm[k]
        for i in range(len(pts)):
            px, py = pts[i]
            A[row, 0] = -nx; A[row, 1] = -ny; A[row, 2] = -1.0
            b[row] = -(px * nx + py * ny); row += 1
            A[row, 0] = nx; A[row, 1] = ny; A[row, 2] = -1.0
            b[row] = px * nx + py * ny; row += 1
    bounds = [(None, None), (None, None), (0, None)]
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            D = float(res.x[2])
            S = (D + APOTHEM) / APOTHEM
            return S, float(res.x[0]), float(res.x[1]), D
    except Exception:
        pass
    return float('inf'), None, None, None


def _soft_max(vals, t):
    m = np.max(vals)
    if not np.isfinite(m):
        return 0.0
    return m + (1.0 / t) * math.log(np.sum(np.exp(t * (vals - m))))


def _smooth_D(pts, beta, cx, cy, t):
    """Smooth surrogate of D = max_k max_i |n_k.(p_i - c)| via log-sum-exp."""
    nrm = _normals(beta)                       # (6,2)
    centered = pts - np.array([cx, cy])        # (n,2)
    proj = centered @ nrm.T                    # (n,6)
    allp = np.concatenate([proj.ravel(), (-proj).ravel()])
    return _soft_max(allp, t)


def _smooth_S(pts, beta, cx, cy, t):
    return (_smooth_D(pts, beta, cx, cy, t) + APOTHEM) / APOTHEM


def _smooth_obj_grad(pts, beta, cx, cy, t):
    """Return (dbeta, dcx, dcy) of the soft-max D surrogate."""
    nrm = _normals(beta)
    centered = pts - np.array([cx, cy])
    proj = centered @ nrm.T
    allp = np.concatenate([proj.ravel(), (-proj).ravel()])
    m = np.max(allp)
    e = np.exp(t * (allp - m))
    w = e / np.sum(e)
    half = len(allp) // 2
    nrmx = np.tile(nrm[:, 0], len(pts))
    nrmy = np.tile(nrm[:, 1], len(pts))
    dcx = -np.dot(w, nrmx)
    dcy = -np.dot(w, nrmy)
    ddbeta = np.zeros(len(allp))
    for k in range(6):
        idx = slice(k * len(pts), (k + 1) * len(pts))
        ddbeta[idx] = (nrm[k, 1] * (pts[:, 0] - cx) - nrm[k, 0] * (pts[:, 1] - cy))
    ddbeta[half:] = -ddbeta[:half]
    dbeta = np.dot(w, ddbeta)
    return dbeta, dcx, dcy


def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = [
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (1, 1), (-1, 2), (-2, 1), (2, -1)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 1), (-2, 2), (-1, 3), (1, -2)],
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
         (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (1, 3), (2, 3)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (1, 1), (-1, -1)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (1, -2)],
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 1), (1, 2), (-1, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2)],
        [(0, 0), (1, 0), (2, 0),
         (1, 1), (2, 1), (3, 1), (4, 1),
         (1, 2), (2, 2), (3, 2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1),
         (0, 2), (1, 2),
         (0, 3), (1, 3),
         (0, 4)],
    ]

    best_S = float('inf')
    best_result = None

    for axial in candidates:
        if not _time_ok():
            break
        pts = _axial_to_cartesian(axial)

        # Phase 1: coarse grid over beta, exact LP
        coarse = np.linspace(0.0, SIXTH_PI, 25)
        cvals = np.array([_solve_lp(pts, float(b))[0] for b in coarse])
        i0 = int(np.argmin(cvals))
        S0 = float(cvals[i0])
        if S0 == float('inf'):
            continue
        _, cx0, cy0, _ = _solve_lp(pts, float(coarse[i0]))
        beta0 = float(coarse[i0])

        refined = (S0, cx0, cy0, beta0)

        # Phase 2: smooth (beta, cx, cy) refinement via soft-max + L-BFGS-B
        try:
            if _time_ok():
                x0 = np.array([cx0, cy0, beta0])

                def obj(x):
                    cx, cy, beta = x
                    return _smooth_S(pts, beta, cx, cy, 8.0)

                def obj_grad(x):
                    cx, cy, beta = x
                    db, dx, dy = _smooth_obj_grad(pts, beta, cx, cy, 8.0)
                    return np.array([dx, dy, db])

                res = minimize(obj, x0, jac=obj_grad, method='L-BFGS-B',
                               bounds=[(None, None), (None, None), (0.0, SIXTH_PI)],
                               options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < refined[0]:
                    cxr, cyr, betar = res.x
                    S, cx, cy, D = _solve_lp(pts, betar)
                    if S < refined[0]:
                        refined = (S, cx, cy, betar)
        except Exception:
            pass

        # Phase 3: high-precision polish of beta with exact LP objective
        try:
            if _time_ok():
                S, cx, cy, D = _solve_lp(pts, refined[3])
                if S < refined[0]:
                    refined = (S, cx, cy, refined[3])

                def lp_obj(beta):
                    return _solve_lp(pts, beta)[0]

                for (lo, hi, it, xt) in [
                    (max(0.0, refined[3] - 0.02), min(SIXTH_PI, refined[3] + 0.02), 400, 1e-10),
                    (max(0.0, refined[3] - 0.003), min(SIXTH_PI, refined[3] + 0.003), 800, 1e-13),
                    (max(0.0, refined[3] - 0.0005), min(SIXTH_PI, refined[3] + 0.0005), 1200, 1e-15),
                ]:
                    if not _time_ok():
                        break
                    r = minimize_scalar(lp_obj, bounds=(lo, hi), method='bounded',
                                        options={'maxiter': it, 'xatol': xt})
                    if r.fun < refined[0]:
                        S, cx, cy, D = _solve_lp(pts, float(r.x))
                        refined = (S, cx, cy, float(r.x))
        except Exception:
            pass

        if refined[0] < best_S:
            best_S = refined[0]
            best_result = (refined[1], refined[2], refined[3], pts.copy())

    if best_result is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    cx, cy, beta, pts = best_result
    S, cx, cy, D = _solve_lp(pts, beta)
    if S == float('inf'):
        S = best_S
    angle_deg = math.degrees(beta)
    inner = [[float(pts[i, 0]), float(pts[i, 1]), angle_deg] for i in range(len(pts))]
    return inner, [float(cx), float(cy)], float(S), angle_deg
