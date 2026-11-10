# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _build_sofa_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return gx.ravel(), gy.ravel()


_SX, _SY = _build_sofa_grid()
_N = _SX.size


def _densify(path):
    full = [path[0]]
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        m = max(1, int(np.ceil(abs(p2[2] - p1[2]))))
        for k in range(1, m + 1):
            t = k / m
            full.append(p1 + t * (p2 - p1))
    return np.array(full)


def _eval_path_fast(path):
    placements = np.concatenate([
        np.array([[0.0, 0.0, 0.0]]),
        path,
        np.array([[-2.0, 0.0, 90.0]]),
    ], axis=0)
    placements = _densify(placements)

    angles = np.deg2rad(placements[:, 2])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    xs = placements[:, 0]
    ys = placements[:, 1]

    chunk_size = 50000
    n_chunks = (_N + chunk_size - 1) // chunk_size
    inside = np.ones(_N, dtype=bool)

    for c in range(n_chunks):
        s = c * chunk_size
        e = min(s + chunk_size, _N)
        wx = _SX[s:e]
        wy = _SY[s:e]
        chunk_inside = np.ones(e - s, dtype=bool)
        for pi in range(len(placements)):
            ca = cos_a[pi]
            sa = sin_a[pi]
            x = xs[pi]
            y = ys[pi]
            u = ca * (wx - x) + sa * (wy - y)
            v = -sa * (wx - x) + ca * (wy - y)
            c1 = (u >= -1e-9) & (u <= 1.0 + 1e-9) & (v >= -3.0 - 1e-9) & (v <= 1.0 + 1e-9)
            c2 = (u >= -3.0 - 1e-9) & (u <= 1.0 + 1e-9) & (v >= -1e-9) & (v <= 1.0 + 1e-9)
            in_base = c1 | c2
            chunk_inside &= in_base
            if not np.any(chunk_inside):
                break
        inside[s:e] = chunk_inside
        if not np.any(inside):
            return 0.0

    return float(np.count_nonzero(inside)) * 0.0001


def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    deadline = t_start + 560.0

    n_path = 8
    ts = np.linspace(0.1, 0.9, n_path)
    xs0 = -2.0 * ts
    ys0 = 0.3 * np.sin(np.pi * ts)
    angles0 = np.linspace(10, 80, n_path)

    best_params = np.column_stack([xs0, ys0, angles0]).ravel()
    best_area = _eval_path_fast(np.column_stack([xs0, ys0, angles0]))

    def obj(p):
        if time.time() > deadline:
            return 0.0
        path = p.reshape(n_path, 3)
        return -_eval_path_fast(path)

    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        if restart == 0:
            p0 = best_params.copy()
        else:
            p0 = best_params + np.random.normal(0, 0.15, size=n_path * 3)
            p0[2::3] = np.clip(p0[2::3], 1, 89)
            p0[0::3] = np.clip(p0[0::3], -3, 1)
            p0[1::3] = np.clip(p0[1::3], -1.5, 1.5)

        options = {
            'maxiter': 400,
            'maxfev': 300,
            'xatol': 1e-4,
            'fatol': 1e-5,
            'adaptive': True,
        }
        try:
            res = minimize(obj, p0, method='Nelder-Mead', options=options)
            if res.fun < 0:
                area = -res.fun
                if area > best_area:
                    best_area = area
                    best_params = res.x.copy()
        except Exception:
            pass

    path = best_params.reshape(n_path, 3)
    if not np.all(np.isfinite(path)):
        path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ])
    return path


# EVOLVE-BLOCK-END
