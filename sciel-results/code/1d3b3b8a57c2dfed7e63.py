# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_N = 64

def _make_yw(m):
    y = (np.arange(m) + 0.5) / m
    w = (1.0 - y) / _N
    return y.astype(np.float64), w.astype(np.float64)

_A = ((np.arange(_N) + 1.0) / _N).astype(np.float64)
_Y, _W = _make_yw(8000)

def _set_m(m):
    global _Y, _W
    _Y, _W = _make_yw(m)

def _area_and_grad(x):
    L = x[None, :] + _A[None, :] * _Y[:, None]
    perm = np.argsort(L, axis=1)
    Ls = np.take_along_axis(L, perm, axis=1)
    gaps = Ls[:, 1:] - Ls[:, :-1]
    wcol = _W[:, None]
    union = _W[:, None] + np.minimum(gaps, wcol).sum(axis=1)
    area = float(union.mean())
    after_flags = (gaps < wcol)
    before_flags = np.zeros_like(after_flags)
    before_flags[:, 1:] = after_flags[:, :-1]
    gs = np.empty_like(L)
    gs[:, :_N - 1] = before_flags[:, :_N - 1] - after_flags
    gs[:, _N - 1] = before_flags[:, _N - 1]
    grad = np.zeros_like(L)
    np.put_along_axis(grad, perm, gs, axis=1)
    g = grad.mean(axis=0)
    return area, g

def _keich():
    n = 64
    k = 6
    j = np.arange(1, k + 1)
    pos = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        pos.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(pos, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 540.0
    x_base = _keich()

    def _v_of(x):
        g = x - x[0]
        return g[1:].astype(np.float64)

    def _x_of(v):
        x = np.empty(_N, dtype=np.float64)
        x[0] = 0.0
        x[1:] = v
        return x

    def _f(v):
        a, _ = _area_and_grad(_x_of(v))
        return a

    def _g(v):
        _, gr = _area_and_grad(_x_of(v))
        return gr[1:]

    _set_m(20000)
    best = _x_of(_v_of(x_base))
    best_a = _area_and_grad(best)[0]
    _set_m(8000)

    rng = np.random.default_rng(20240517)
    starts = [
        _v_of(x_base),
        np.zeros(_N - 1),
        _v_of(-0.5 * _A),
        _v_of(-_A),
        _v_of(0.5 * _A),
    ]
    for _ in range(6):
        starts.append(rng.normal(0.0, 0.1, _N - 1))

    it = 0
    try:
        while time.time() < deadline - 130.0:
            it += 1
            if it <= len(starts):
                v0 = starts[it - 1]
            else:
                scale = 0.05 if (it % 12) != 0 else 0.25
                v0 = best[1:] + rng.normal(0.0, scale, _N - 1)
            try:
                res = minimize(_f, v0, jac=_g, method="L-BFGS-B",
                               options={"maxiter": 250, "gtol": 1e-8, "maxls": 50})
                v = np.asarray(res.x, dtype=np.float64)
                x = _x_of(v)
                _set_m(20000)
                a = _area_and_grad(x)[0]
                _set_m(8000)
                if np.isfinite(a) and a < best_a:
                    best_a = a
                    best = x
            except Exception:
                pass
    except Exception:
        pass

    # final polish at full resolution
    try:
        _set_m(20000)
        v0 = best[1:]
        res = minimize(_f, v0, jac=_g, method="L-BFGS-B",
                       options={"maxiter": 2500, "gtol": 1e-11, "maxls": 50})
        x = _x_of(np.asarray(res.x, dtype=np.float64))
        a = _area_and_grad(x)[0]
        if np.isfinite(a) and a < best_a:
            best_a = a
            best = x
    except Exception:
        pass

    out = np.asarray(best, dtype=np.float64)
    if out.shape != (_N,) or not np.all(np.isfinite(out)):
        out = x_base
    # sanity: area > 0
    _set_m(20000)
    try:
        a = _area_and_grad(out)[0]
        if not (np.isfinite(a) and a > 0):
            out = x_base
    except Exception:
        out = x_base
    return out
# EVOLVE-BLOCK-END
