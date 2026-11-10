# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_length_vec(starts, widths):
    """
    Compute total length of union of 1-D intervals [starts[i], starts[i]+widths[i]].
    Returns (length, grad) where grad is the gradient w.r.t. starts.
    Uses a differentiable approach via sorting and merge.
    """
    n = len(starts)
    if n == 0:
        return 0.0, np.zeros(0)
    
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = s_sorted + widths[order]
    
    # Merge intervals: compute union length
    # We'll compute the union length and its gradient via a sweep.
    # For each sorted interval, determine how much it contributes.
    
    # First, compute the union length using standard merge
    total_len = 0.0
    cs = s_sorted[0]
    ce = e_sorted[0]
    contrib = np.zeros(n)  # contribution of each interval in sorted order
    
    for i in range(1, n):
        if s_sorted[i] > ce:
            total_len += ce - cs
            contrib[0] = ce - cs
            cs = s_sorted[i]
            ce = e_sorted[i]
        else:
            ce = max(ce, e_sorted[i])
    
    total_len += ce - cs
    contrib[0] = ce - cs  # this is approximate; we need a better gradient
    
    # For gradient computation, use a simpler approach:
    # The union length is piecewise linear in starts.
    # We can compute it by noting that union length = sum of lengths - sum of overlaps + ...
    # But that's complex. Instead, use numerical differentiation or a smarter formula.
    
    # Actually, for optimization, let's just use the value and compute gradient numerically
    # or use a different approach. Let's return just the value and use finite differences
    # in the optimizer, or better yet, compute the gradient analytically.
    
    # Simpler: compute union length and use scipy's finite difference gradient
    return total_len


def _exact_volume(pos_x, pos_y, N=8, c_samples=200):
    """Exact union volume via product structure: integral of Lx(c)*Ly(c) dc."""
    c_vals = np.linspace(0, 1, c_samples)
    vol = 0.0
    invN = 1.0 / N
    dx = 1.0 / c_samples
    
    for c in c_vals:
        # Compute x intervals
        x_starts = pos_x + c * np.arange(N) * invN
        x_widths = np.full(N, invN)
        
        # Compute y intervals
        y_starts = pos_y + c * np.arange(N) * invN
        y_widths = np.full(N, invN)
        
        # Union length in x
        x_order = np.argsort(x_starts)
        x_s = x_starts[x_order]
        x_e = x_s + x_widths[x_order]
        
        lx = 0.0
        if len(x_s) > 0:
            cs = x_s[0]
            ce = x_e[0]
            for i in range(1, len(x_s)):
                if x_s[i] > ce:
                    lx += ce - cs
                    cs = x_s[i]
                    ce = x_e[i]
                else:
                    ce = max(ce, x_e[i])
            lx += ce - cs
        
        # Union length in y
        y_order = np.argsort(y_starts)
        y_s = y_starts[y_order]
        y_e = y_s + y_widths[y_order]
        
        ly = 0.0
        if len(y_s) > 0:
            cs = y_s[0]
            ce = y_e[0]
            for i in range(1, len(y_s)):
                if y_s[i] > ce:
                    ly += ce - cs
                    cs = y_s[i]
                    ce = y_e[i]
                else:
                    ce = max(ce, y_e[i])
            ly += ce - cs
        
        vol += lx * ly
    
    return vol * dx


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 570.0  # Safety margin

    # Keich binary construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)

    best_x = keich.copy()
    best_y = keich.copy()
    best_vol = _exact_volume(best_x, best_y, N, c_samples=200)

    def objective(params):
        if time.time() > deadline - 30:
            return best_vol
        return _exact_volume(params[:N], params[N:], N, c_samples=200)

    options = {'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
    rng = np.random.default_rng(12345)

    n_restarts = 0
    while time.time() < deadline - 60:
        scale = max(0.003, 0.2 * best_vol ** 0.5)
        init = np.concatenate([best_x, best_y]) + rng.normal(0, scale, 2 * N)
        result = minimize(objective, init, method='Nelder-Mead', options=options)
        if result.fun < best_vol:
            best_vol = result.fun
            best_x = result.x[:N].copy()
            best_y = result.x[N:].copy()
        n_restarts += 1

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
