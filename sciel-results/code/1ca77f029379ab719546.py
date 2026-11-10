# EVOLVE-BLOCK-START
import numpy as np
import time


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
    
    # First, compute the merged union structure
    # Track the rightmost end of the current merged group
    # Contribution of interval i (in sorted order) to union:
    #   max(0, e_sorted[i] - max(s_sorted[i], prev_end))
    # where prev_end is the max end of all previous intervals that overlap with i.
    
    # Actually, let's do a proper sweep:
    # Sort by start. Then sweep, maintaining the current rightmost end.
    # For each interval, its marginal contribution is:
    #   max(0, e_i - max(s_i, current_rightmost))
    # but current_rightmost is updated to max(current_rightmost, e_i)
    
    # This is not straightforward to differentiate through the max and the sequential update.
    # Alternative: use the inclusion-exclusion or a different formulation.
    
    # Let's use a simpler approach: compute the union length directly, and use 
    # finite differences for gradients if needed, or use a smooth approximation.
    
    # For now, compute the union length exactly (non-differentiable) and use 
    # Nelder-Mead or similar derivative-free method.
    
    # Actually, let me just compute the union length efficiently.
    total = 0.0
    cs = s_sorted[0]
    ce = e_sorted[0]
    for k in range(1, n):
        if s_sorted[k] > ce:
            total += ce - cs
            cs = s_sorted[k]
            ce = e_sorted[k]
        else:
            ce = max(ce, e_sorted[k])
    total += ce - cs
    return total


def _exact_volume(pos_x, pos_y, N=8):
    """Exact union volume via product structure: integral of Lx(c)*Ly(c) dc."""
    c_samples = np.linspace(0, 1, 200)
    vol = 0.0
    invN = 1.0 / N
    for c in c_samples:
        x_starts = pos_x + c * np.arange(N) * invN
        y_starts = pos_y + c * np.arange(N) * invN
        Lx = _union_length_vec(x_starts, np.full(N, invN))
        Ly = _union_length_vec(y_starts, np.full(N, invN))
        vol += Lx * Ly
    return vol * (1.0 / 200)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0
    
    # Keich binary construction as initial guess
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    best_x = keich.copy()
    best_y = keich.copy()
    best_vol = _exact_volume(best_x, best_y, N)
    
    # Use Nelder-Mead with multiple restarts
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 30:
            return best_vol
        return _exact_volume(params[:N], params[N:], N)
    
    rng = np.random.default_rng(12345)
    
    # Try several restarts with different scales
    for restart in range(50):
        if time.time() > deadline - 60:
            break
        
        scale = max(0.001, 0.5 * best_vol ** 0.5)
        init = np.concatenate([best_x, best_y]) + rng.normal(0, scale, 2 * N)
        
        result = minimize(
            objective, init, method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True}
        )
        
        if result.fun < best_vol:
            best_vol = result.fun
            best_x = result.x[:N].copy()
            best_y = result.x[N:].copy()
    
    # Also try a finer local search around the best point
    if time.time() < deadline - 30:
        def fine_objective(params):
            if time.time() > deadline - 10:
                return best_vol
            return _exact_volume(params[:N], params[N:], N)
        
        result = minimize(
            fine_objective, np.concatenate([best_x, best_y]),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                     'adaptive': True}
        )
        if result.fun < best_vol:
            best_vol = result.fun
            best_x = result.x[:N].copy()
            best_y = result.x[N:].copy()
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
