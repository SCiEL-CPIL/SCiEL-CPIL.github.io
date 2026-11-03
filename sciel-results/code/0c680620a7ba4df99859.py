import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _autoconv_peak(h):
    """Supremum of continuous autoconvolution of step function with heights h on equal intervals in [-1/4,1/4]."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    d = 0.5 / n  # interval width: total length 0.5, n intervals
    # centers of intervals
    x = -0.25 + d * (np.arange(n) + 0.5)
    # autoconvolution (f*f)(t) = integral f(s) f(t-s) ds
    # evaluate on a fine grid of t in [-0.5, 0.5]
    t = np.linspace(-0.5, 0.5, 4001)
    # f(s) for s in [-0.25,0.25]; f(t-s) nonzero when t-s in [-0.25,0.25]
    # Use piecewise: for each t, integral over s of f(s)f(t-s)
    # Vectorize: build f on a fine s-grid
    s = np.linspace(-0.25, 0.25, 4001)
    ds = s[1] - s[0]
    # f_s = h for s in interval i
    idx_s = np.clip(((s + 0.25) / d).astype(int), 0, n - 1)
    f_s = h[idx_s]
    # for each t, f(t-s): t-s in [-0.25,0.25]
    # f_ts at s: value h[j] where t-s in interval j
    idx_ts = np.clip(((t[:, None] - s[None, :] + 0.25) / d).astype(int), 0, n - 1)
    # but only where t-s actually in [-0.25,0.25]
    ts = t[:, None] - s[None, :]
    mask = (ts >= -0.25) & (ts <= 0.25)
    f_ts = np.where(mask, h[idx_ts], 0.0)
    conv = (f_s[None, :] * f_ts).sum(axis=1) * ds
    return float(conv.max())

def _ratio(h):
    h = np.asarray(h, dtype=float)
    s1 = h.sum()
    if s1 <= 0:
        return 0.0
    s2 = (h ** 2).sum()
    peak = _autoconv_peak(h)
    if peak <= 0:
        return 0.0
    return s2 / (s1 * peak)

def _grad_peak(h):
    """Analytic-ish gradient of peak w.r.t. h via finite diff on the active t (subgradient)."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    d = 0.5 / n
    s = np.linspace(-0.25, 0.25, 4001)
    ds = s[1] - s[0]
    idx_s = np.clip(((s + 0.25) / d).astype(int), 0, n - 1)
    f_s = h[idx_s]
    t = np.linspace(-0.5, 0.5, 4001)
    idx_ts = np.clip(((t[:, None] - s[None, :] + 0.25) / d).astype(int), 0, n - 1)
    ts = t[:, None] - s[None, :]
    mask = (ts >= -0.25) & (ts <= 0.25)
    f_ts = np.where(mask, h[idx_ts], 0.0)
    conv = (f_s[None, :] * f_ts).sum(axis=1) * ds
    tstar = t[int(np.argmax(conv))]
    # gradient of conv(tstar) w.r.t. h: d/dh_i [sum_s f_s(s) f_ts(s) ds]
    # f_s(s) = h[idx_s(s)], f_ts(s) = h[idx_ts(s)] (where mask)
    # d conv/dh_i = sum_s [ (d f_s/dh_i) f_ts + f_s (d f_ts/dh_i) ] ds
    # = sum_s [ 1_{idx_s(s)=i} f_ts(s) + f_s(s) 1_{idx_ts(s)=i} ] ds
    g = np.zeros(n)
    for i in range(n):
        a = (idx_s == i) & mask
        b = (idx_ts == i) & mask
        g[i] = (f_ts[a].sum() + f_s[b].sum()) * ds
    return g

def construct():
    global DEADLINE
    best = np.ones(N)
    best_val = _ratio(best)

    def timeup():
        return time.time() > DEADLINE

    # SLSQP: maximize ratio by minimizing -ratio, but ratio not smooth; instead
    # minimize peak subject to sum=1, h>=0, and maximize sum h^2 via the objective
    # Use objective: minimize peak / sum h^2  (equivalent to maximizing ratio with sum=1)
    def obj(x):
        s2 = (x ** 2).sum()
        if s2 <= 1e-12:
            return 1e12
        return _autoconv_peak(x) / s2

    def obj_grad(x):
        s2 = (x ** 2).sum()
        if s2 <= 1e-12:
            return np.zeros(N)
        gp = _grad_peak(x)
        # d (peak/s2)/dx = gp/s2 - peak * 2x / s2^2
        peak = _autoconv_peak(x)
        return gp / s2 - peak * 2.0 * x / (s2 * s2)

    cons = [{'type': 'eq', 'fun': lambda x: x.sum() - 1.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, None)] * N

    # multiple restarts
    rng = np.random.default_rng(0)
    n_restarts = 6
    for r in range(n_restarts):
        if timeup():
            break
        if r == 0:
            x0 = np.ones(N)
        elif r == 1:
            x0 = np.zeros(N); x0[0] = 1.0
        elif r == 2:
            x0 = np.zeros(N); x0[N//2] = 1.0
        elif r == 3:
            x0 = np.zeros(N); x0[:N//2] = 1.0
            x0 /= x0.sum()
        else:
            x0 = rng.dirichlet(np.ones(N))
        try:
            res = minimize(obj, x0, jac=obj_grad, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 80, 'ftol': 1e-9})
            if res.success or res.fun < 1e12:
                x = np.clip(res.x, 0, None)
                x /= x.sum()
                v = _ratio(x)
                if v > best_val:
                    best_val = v
                    best = x.copy()
        except Exception:
            pass

    # coordinate descent refinement
    if not timeup():
        h = best.copy()
        improved = True
        it = 0
        while improved and not timeup() and it < 2000:
            improved = False
            it += 1
            cur = _ratio(h)
            for i in range(N):
                if timeup():
                    break
                # try moving mass from i to j
                for j in range(N):
                    if i == j:
                        continue
                    if h[i] < 1e-9:
                        continue
                    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
                        h2 = h.copy()
                        h2[i] -= h[i] * frac
                        h2[j] += h[i] * frac
                        if h2.min() < -1e-12:
                            continue
                        v2 = _ratio(h2)
                        if v2 > cur + 1e-12:
                            h = h2
                            cur = v2
                            improved = True
                            break
        if _ratio(h) > best_val:
            best_val = _ratio(h)
            best = h.copy()

    return [float(x) for x in best]
