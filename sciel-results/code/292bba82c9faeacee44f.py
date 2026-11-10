# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 600.0


def _time_left():
    return _DEADLINE - time.time() - 30.0


def _keich_offsets(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _compute_1d_union_length(positions, N, n_c=80):
    c_vals = np.linspace(0, 1, n_c)
    lengths = np.empty(n_c)
    arange_N = np.arange(N, dtype=np.float64) / N
    inv_N = 1.0 / N
    for idx in range(n_c):
        c = c_vals[idx]
        starts = positions + c * arange_N
        ends = starts + inv_N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        total = 0.0
        cs = s_sorted[0]
        ce = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] > ce:
                total += ce - cs
                cs = s_sorted[k]
                ce = e_sorted[k]
            elif e_sorted[k] > ce:
                ce = e_sorted[k]
        total += ce - cs
        lengths[idx] = total
    return np.trapz(lengths, c_vals)


def _compute_1d_union_length_fast(positions, N, n_c=40):
    c_vals = np.linspace(0, 1, n_c)
    lengths = np.empty(n_c)
    arange_N = np.arange(N, dtype=np.float64) / N
    inv_N = 1.0 / N
    for idx in range(n_c):
        c = c_vals[idx]
        starts = positions + c * arange_N
        ends = starts + inv_N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        total = 0.0
        cs = s_sorted[0]
        ce = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] > ce:
                total += ce - cs
                cs = s_sorted[k]
                ce = e_sorted[k]
            elif e_sorted[k] > ce:
                ce = e_sorted[k]
        total += ce - cs
        lengths[idx] = total
    return np.trapz(lengths, c_vals)


def _evaluate_volume(positions, N):
    i_vals = (np.arange(N * N) // N).astype(np.float64)
    j_vals = (np.arange(N * N) % N).astype(np.float64)
    x_base = positions[:, 0]
    y_base = positions[:, 1]
    x_ext = x_base + 1.0 / N + i_vals / N
    y_ext = y_base + 1.0 / N + j_vals / N
    X0 = float(np.min(x_base))
    X1 = float(np.max(x_ext))
    Y0 = float(np.min(y_base))
    Y1 = float(np.max(y_ext))
    if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
        return np.inf
    rng = np.random.default_rng(12345)
    n_mc = 500000
    a = rng.uniform(X0, X1, n_mc)
    b = rng.uniform(Y0, Y1, n_mc)
    c = rng.uniform(0.0, 1.0, n_mc)
    chunk = 50000
    inside_any = np.empty(n_mc, dtype=bool)
    for s in range(0, n_mc, chunk):
        e = min(s + chunk, n_mc)
        a_c = a[s:e]
        b_c = b[s:e]
        c_c = c[s:e]
        xl = x_base + c_c[:, None] * i_vals[None, :] / N
        xh = xl + 1.0 / N
        yl = y_base + c_c[:, None] * j_vals[None, :] / N
        yh = yl + 1.0 / N
        inx = (a_c[:, None] >= xl) & (a_c[:, None] <= xh)
        iny = (b_c[:, None] >= yl) & (b_c[:, None] <= yh)
        inside_any[s:e] = np.any(inx & iny, axis=1)
    frac = float(np.mean(inside_any))
    return frac * (X1 - X0) * (Y1 - Y0)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    keich = _keich_offsets(N)
    
    best_x = keich.copy()
    best_1d_len = _compute_1d_union_length_fast(keich, N, n_c=50)
    
    rng_r = np.random.default_rng(2024)
    n_restarts = 25
    max_iters = 200
    
    for r in range(n_restarts):
        if _time_left() < 90:
            break
        if r == 0:
            x0 = keich.copy()
        else:
            scale = 0.03 * (1.0 + 0.5 * r / n_restarts)
            x0 = keich + rng_r.normal(0.0, scale, size=N)
        try:
            res = minimize(
                lambda x: _compute_1d_union_length_fast(x, N, n_c=50),
                x0, method='Nelder-Mead',
                options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_1d_len:
                best_1d_len = float(res.fun)
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Fine-tune with more integration points
    if _time_left() > 60:
        try:
            res2 = minimize(
                lambda x: _compute_1d_union_length_fast(x, N, n_c=100),
                best_x, method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if res2.fun < best_1d_len:
                best_1d_len = float(res2.fun)
                best_x = res2.x.copy()
        except Exception:
            pass
    
    i_vals = np.repeat(np.arange(N), N)
    j_vals = np.tile(np.arange(N), N)
    base_x = best_x[i_vals]
    base_y = best_x[j_vals]
    positions = np.stack([base_x, base_y], axis=1)
    
    V = _evaluate_volume(positions, N)
    if not np.isfinite(V) or V <= 0:
        base_x = keich[i_vals]
        base_y = keich[j_vals]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
