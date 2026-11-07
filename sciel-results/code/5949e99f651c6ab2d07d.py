import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 135.0  # safety margin under 150s kill

def _eval(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(np.abs(g)) / (m + 1)
    ninf = np.max(np.abs(g))
    return l2 / (n1 * ninf)

def _penalized(h):
    h = np.maximum(h, 0.0)
    return -_eval(h)

def _cd(h0, budget):
    h = np.array(h0, dtype=float)
    best = -_penalized(h)
    step = 0.1
    while time.time() < budget:
        improved = False
        for i in range(N):
            for d in (step, -step):
                cand = h.copy()
                cand[i] = max(0.0, cand[i] + d)
                v = -_penalized(cand)
                if v > best + 1e-12:
                    h = cand
                    best = v
                    improved = True
        if not improved:
            step *= 0.6
            if step < 1e-5:
                break
    return h, best

def _nm(h0, budget):
    try:
        res = minimize(_penalized, h0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-7})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            return h, -_penalized(h)
    except Exception:
        pass
    return h0, -_penalized(h0)

def _shapes(t, a):
    out = []
    out.append(np.ones(N))
    out.append(np.maximum(np.cos(np.pi * t), 0.0))
    out.append(np.maximum(1.0 - 2.0 * np.abs(t), 0.0))
    out.append(np.maximum(1.0 - 4.0 * t ** 2, 0.0))
    out.append(np.exp(-a * t ** 2))
    out.append(np.exp(-a * np.abs(t)))
    out.append(np.maximum(1.0 - a * np.abs(t), 0.0))
    out.append(np.maximum(1.0 - a * t ** 2, 0.0))
    out.append(np.maximum(1.0 - a ** 2 * t ** 2, 0.0))
    out.append(np.maximum(1.0 - a * t ** 4, 0.0))
    out.append((1.0 - t ** 2) ** a)
    out.append(np.sin(np.pi * t) ** a)
    out.append(np.maximum(1.0 - a * t ** 2 - a * t, 0.0))
    return out

def construct():
    t = np.linspace(-0.5, 0.5, N)
    # 1) analytic shape exploration
    best_h = np.ones(N)
    best_v = _eval(best_h)
    for a in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        for sh in _shapes(t, a):
            if not np.all(np.isfinite(sh)) or np.min(sh) < -1e-9:
                continue
            sh = np.maximum(sh, 0.0)
            if np.max(sh) <= 0:
                continue
            v = _eval(sh)
            if v > best_v:
                best_v = v
                best_h = sh.copy()
    # 2) coordinate descent refinement
    budget = min(time.time() + 55.0, _DEADLINE - 65.0)
    h, v = _cd(best_h, budget)
    if v > best_v:
        best_v, best_h = v, h
    # 3) Nelder-Mead polish
    budget2 = min(time.time() + 55.0, _DEADLINE - 10.0)
    h2, v2 = _nm(best_h, budget2)
    if v2 > best_v:
        best_v, best_h = v2, h2
    # 4) final coordinate-descent pass
    budget3 = min(time.time() + 30.0, _DEADLINE - 5.0)
    h3, v3 = _cd(best_h, budget3)
    if v3 > best_v:
        best_v, best_h = v3, h3
    # deadline guard: return best found
    final = np.maximum(best_h, 0.0)
    final = np.where(np.isfinite(final), final, 0.0)
    return final.tolist()
