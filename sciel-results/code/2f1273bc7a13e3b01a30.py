# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses Linear Programming to find a good initial configuration by minimizing
    the width of the line arrangement, followed by local refinement.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    
    # Use a subset of heights for the LP formulation and optimization
    # 1000 points is a good balance between accuracy and speed
    M = 1000
    y_subset = (np.arange(M) + 0.5) / 20000.0
    
    # Objective for LP: Minimize sum of (u_m - v_m)
    # Variables: x (n), u (M), v (M)
    # Order: x_0...x_{n-1}, u_0...u_{M-1}, v_0...v_{M-1}
    c = np.zeros(n + 2 * M)
    # Minimize sum(u_m - v_m) -> c for u is 1/M, c for v is -1/M
    # Actually, we want to minimize the integral, which is approx mean.
    # So minimize sum(u_m - v_m) / M.
    c[n:n+M] = 1.0 / M
    c[n+M:n+2*M] = -1.0 / M
    
    # Constraints:
    # u_m >= x_i + a_i * y_m  =>  -x_i - a_i*y_m + u_m >= 0  =>  x_i + a_i*y_m - u_m <= 0
    # v_m <= x_i + a_i * y_m  =>  -x_i - a_i*y_m + v_m <= 0
    # We need A_ub such that A_ub @ var <= b_ub
    
    # Number of constraints: 2 * n * M
    n_constraints = 2 * n * M
    A_ub = np.zeros((n_constraints, n + 2 * M))
    b_ub = np.zeros(n_constraints)
    
    # Fill A_ub
    # For each m, for each i:
    # Row 1: x_i + a_i*y_m - u_m <= 0
    # Row 2: -x_i - a_i*y_m + v_m <= 0
    
    for m in range(M):
        y_m = y_subset[m]
        for i in range(n):
            # Constraint 1: x_i + a_i*y_m - u_m <= 0
            idx1 = 2 * (m * n + i)
            A_ub[idx1, i] = 1.0
            A_ub[idx1, n + m] = -1.0
            
            # Constraint 2: -x_i - a_i*y_m + v_m <= 0
            idx2 = idx1 + 1
            A_ub[idx2, i] = -1.0
            A_ub[idx2, n + M + m] = 1.0
    
    # Bounds: x can be anything (unbounded), u, v can be anything.
    # However, to prevent numerical issues, we can bound x reasonably.
    # The optimal x should be within [-1, 1] or so.
    bounds = [(-10.0, 10.0)] * n + [(None, None)] * (2 * M)
    
    # Solve LP
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res_lp.success:
        x_init = res_lp.x[:n]
    else:
        # Fallback to simple initialization
        x_init = np.zeros(n)
        
    # Refine with local optimization using the true objective
    # Use a subset of heights for speed during optimization
    m_opt = np.arange(0, 20000, 10) # 2000 heights
    y_opt = (m_opt + 0.5) / 20000.0
    
    def area_objective(x):
        """Compute union area using subset of heights for speed."""
        # L_i = x_i + a_i * y for each y in y_opt
        L = x[None, :] + a[None, :] * y_opt[:, None]
        L_sorted = np.sort(L, axis=1)
        w = (1 - y_opt) / n
        diffs = np.diff(L_sorted, axis=1)
        mins = np.minimum(diffs, w[:, None])
        union_lengths = w + np.sum(mins, axis=1)
        return np.mean(union_lengths)
    
    # Use Nelder-Mead which is often faster than Powell for this size
    result = minimize(
        area_objective,
        x_init,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True
        }
    )
    
    x_opt = result.x
    
    # Verify with full 20000 heights
    m_full = np.arange(20000)
    y_full = (m_full + 0.5) / 20000.0
    L_full = x_opt[None, :] + a[None, :] * y_full[:, None]
    L_sorted = np.sort(L_full, axis=1)
    w_full = (1 - y_full) / n
    diffs_full = np.diff(L_sorted, axis=1)
    mins_full = np.minimum(diffs_full, w_full[:, None])
    union_lengths_full = w_full + np.sum(mins_full, axis=1)
    final_area = np.mean(union_lengths_full)
    
    if not np.isfinite(final_area) or final_area <= 0:
        return x_init
    
    return x_opt.astype(np.float64)

# EVOLVE-BLOCK-END
