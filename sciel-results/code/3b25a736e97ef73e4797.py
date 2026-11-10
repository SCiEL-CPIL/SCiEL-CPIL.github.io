import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.arange(1, n + 1, dtype=np.float64) / n
    inv_n = 1.0 / n

    # ---- exact metric (evaluator formula) ----
    def exact_area(x, num_h=20000):
        ys = (np.arange(num_h, dtype=np.float64) + 0.5) / num_h
        ws = (1.0 - ys) * inv_n
        L = x[None, :] + a[None, :] * ys[:, None]
        Ls = np.sort(L, axis=1)
        diffs = np.diff(Ls, axis=1)
        ext = np.minimum(ws[:, None], diffs)
        return float((ws + ext.sum(axis=1)).mean())

    # ---- smooth surrogate (log-sum-exp / soft-min) ----
    def soft_union_len(L, ws, beta):
        diffs = np.diff(L, axis=1)
        logw = np.log(ws)
        loga = np.log(diffs + 1e-15)
        m = np.maximum(logw, loga)
        lse = m + np.log(np.exp(logw - m) + np.exp(loga - m))
        return np.exp(lse).sum(axis=1) + ws

    def smooth_area(x, num_h=1000, beta=120.0):
        ys = (np.arange(num_h, dtype=np.float64) + 0.5) / num_h
        ws = (1.0 - ys) * inv_n
        L = x[None, :] + a[None, :] * ys[:, None]
        Ls = np.sort(L, axis=1)
        return float(soft_union_len(Ls, ws, beta).mean())

    def make_smooth_obj(num_h, beta):
        def obj(x):
            return smooth_area(x, num_h, beta)
        return obj

    def keich_seed():
        k = int(np.log2(n))
        j = np.arange(1, k + 1, dtype=np.float64)
        pos = np.zeros(n, dtype=np.float64)
        for i in range(n):
            bits = np.array(list(str(bin(i))[2:]), dtype=np.float64)
            bits = np.pad(bits, (k - len(bits), 0), constant_values=0)
            pos[i] = float(np.sum((1.0 - j) * bits * (2.0 ** -j) / k))
        return pos

    start = time.time()
    deadline = start + 615.0
    rng = np.random.default_rng(0)

    base = keich_seed()
    starts = [base.copy()]
    for s in (1.0, 0.6, 0.3):
        starts.append(base + s * rng.normal(0.0, 0.02, n))

    best_x = base.copy()
    best_exact = exact_area(base)

    # ---- smooth L-BFGS-B from multiple starts ----
    for x0 in starts:
        if time.time() > deadline - 10:
            break
        for num_h, beta in ((1000, 120.0), (1000, 300.0)):
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(make_smooth_obj(num_h, beta), x0, method='L-BFGS-B',
                               options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-9,
                                        'maxls': 50, 'maxfun': 200000})
                if res.success or res.x is not None:
                    if np.all(np.isfinite(res.x)):
                        ea = exact_area(res.x)
                        if ea < best_exact:
                            best_exact = ea
                            best_x = res.x.copy()
                        x0 = res.x.copy()
            except Exception:
                pass

    # ---- exact coordinate-descent polish ----
    def cd_polish(x, budget):
        x = x.copy()
        cur = exact_area(x)
        step = 0.02
        for _ in range(200):
            if time.time() > budget:
                break
            improved = False
            for i in range(n):
                if time.time() > budget:
                    break
                best_v = cur
                best_d = 0.0
                for d in (step, -step):
                    x[i] += d
                    v = exact_area(x)
                    x[i] -= d
                    if v < best_v - 1e-15:
                        best_v = v
                        best_d = d
                if best_d != 0.0:
                    x[i] += best_d
                    cur = best_v
                    improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
        return x, cur

    if time.time() < deadline - 3:
        x, val = cd_polish(best_x, deadline - 3)
        if val < best_exact:
            best_exact = val
            best_x = x

    out = np.asarray(best_x, dtype=np.float64).ravel()
    if not (np.all(np.isfinite(out)) and out.size == n):
        out = base.copy()
    return out
