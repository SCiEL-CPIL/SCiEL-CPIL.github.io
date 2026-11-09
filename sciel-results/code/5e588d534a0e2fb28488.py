# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _base_inside(u, v):
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return horiz | vert


def _build_grid(step):
    xs = np.arange(-3.0, 1.5, step)
    ys = np.arange(-1.5, 1.5, step)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y


def _eval_count(path, X, Y):
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1, a2 = p[i, 2], p[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        if m > 1:
            ts = np.linspace(0.0, 1.0, m + 1)
            for t in ts[1:-1]:
                dense.append(p[i] * (1.0 - t) + p[i + 1] * t)
        dense.append(p[i + 1])
    dense = np.array(dense)

    n_pts = X.size
    inside = np.ones(n_pts, dtype=bool)
    for (cx, cy, a_deg) in dense:
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        dx = X - cx
        dy = Y - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        inside &= _base_inside(u, v)
        if not inside.any():
            break
    return int(inside.sum())


def _initial_path():
    return np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)


def _nelder_mead(f, x0, max_iter=400, step=0.05, tol=1e-7):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 40:
            break
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if abs(vals[0] - vals[-1]) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
    best_i = int(np.argmin(vals))
    return simplex[best_i], vals[best_i]


def moving_sofa_2d() -> np.ndarray:
    best_path = _initial_path()

    # Coarse grid for fast optimization
    Xc, Yc = _build_grid(0.02)

    def make_f(angles):
        n = len(angles)
        def f(flat):
            path = np.empty((n, 3))
            path[:, 0] = flat[0::2]
            path[:, 1] = flat[1::2]
            path[:, 2] = angles
            return -_eval_count(path, Xc, Yc)
        return f

    def optimize(angles, xy0):
        n = len(angles)
        f = make_f(angles)
        xopt, fopt = _nelder_mead(f, xy0, max_iter=350, step=0.05)
        path = np.empty((n, 3))
        path[:, 0] = xopt[0::2]
        path[:, 1] = xopt[1::2]
        path[:, 2] = angles
        return path

    def resample_xy(ref_path, angles):
        n = len(angles)
        xy = np.zeros(2 * n)
        for k, a in enumerate(angles):
            idx = int(np.argmin(np.abs(ref_path[:, 2] - a)))
            xy[2 * k] = ref_path[idx, 0]
            xy[2 * k + 1] = ref_path[idx, 1]
        return xy

    # Baseline score on coarse grid
    base_score_c = _eval_count(best_path, Xc, Yc)

    angle_sets = [
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
        np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0]),
        np.array([15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]),
    ]

    for angles in angle_sets:
        if _elapsed() > _DEADLINE - 50:
            break
        xy0 = resample_xy(best_path, angles)
        cand = optimize(angles, xy0)
        cand_score_c = _eval_count(cand, Xc, Yc)
        if cand_score_c > base_score_c:
            best_path = cand
            base_score_c = cand_score_c

    # Fine-grid verification / final polish with a short Nelder-Mead on fine grid
    Xf, Yf = _build_grid(0.01)
    fine_score = _eval_count(best_path, Xf, Yf)

    if _elapsed() < _DEADLINE - 80:
        # short fine-grid polish on current best angles
        angles = best_path[:, 2]
        n = len(angles)
        xy0 = np.zeros(2 * n)
        xy0[0::2] = best_path[:, 0]
        xy0[1::2] = best_path[:, 1]

        def f_fine(flat):
            path = np.empty((n, 3))
            path[:, 0] = flat[0::2]
            path[:, 1] = flat[1::2]
            path[:, 2] = angles
            return -_eval_count(path, Xf, Yf)

        xopt, fopt = _nelder_mead(f_fine, xy0, max_iter=80, step=0.02)
        p = np.empty((n, 3))
        p[:, 0] = xopt[0::2]
        p[:, 1] = xopt[1::2]
        p[:, 2] = angles
        s = _eval_count(p, Xf, Yf)
        if s > fine_score:
            best_path = p
            fine_score = s

    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = _initial_path()
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_path()
    return best_path


# EVOLVE-BLOCK-END
