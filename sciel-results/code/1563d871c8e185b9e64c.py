# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    n = len(support)
    s = support.astype(np.float64)
    w = weights.astype(np.float64)
    total_w = w.sum()
    if total_w <= 1e-9:
        return 0.0
    p = w / total_w  # normalized probabilities
    
    # Pairwise sums s_i + s_j for i, j in range(n)
    ss = s[:, None] + s[None, :]  # shape (n, n)
    pp = p[:, None] * p[None, :]  # shape (n, n)
    
    # For each (k, l) pair, we need sum of p_i * p_j over (i,j) with s_i + s_j < 2*s_l - s_k
    # Threshold for (k, l): t_{k,l} = 2*s_l - s_k
    # P = sum_{k,l} p_k * p_l * (sum_{i,j: ss[i,j] < 2*s_l - s_k} pp[i,j])
    
    # This is O(n^4) naively. Let's vectorize over (k,l) pairs.
    # For each (k, l), threshold = 2*s_l - s_k, and we need sum of pp where ss < threshold.
    
    # We can precompute for all possible thresholds. But thresholds depend on k, l.
    # Alternative: for each l, compute the cumulative distribution of pair sums.
    
    # Better approach: sort the pair sums and use cumulative sums.
    ss_flat = ss.ravel()  # n^2 values
    pp_flat = pp.ravel()  # n^2 values
    
    # Sort by ss_flat
    order = np.argsort(ss_flat, kind='mergesort')
    ss_sorted = ss_flat[order]
    pp_sorted = pp_flat[order]
    pp_cumsum = np.cumsum(pp_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Find number of pair sums strictly less than threshold
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            if idx > 0:
                pair_prob = pp_cumsum[idx - 1]
            else:
                pair_prob = 0.0
            P += p[k] * p[l] * pair_prob
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a fixed support grid and solve a linear program to find optimal weights.
    The LP formulation: maximize c^T w subject to w >= 0, sum(w) = 1,
    where c_l = sum over (i,j,k) such that s_i + s_j + s_k < 2*s_l of p_i * p_j * p_k... 
    Actually, the objective is P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2 s_l - s_k} w_i w_j)
    
    This is NOT linear in w. It's a quartic form. So LP doesn't directly apply.
    
    Let me reconsider. The objective P = sum_{k,l,i,j: s_i+s_j < 2 s_l - s_k} w_k w_l w_i w_j
    This is a 4th degree polynomial in weights. Not a linear program.
    
    Alternative approach: Use Nelder-Mead or similar derivative-free optimization
    over both support points and weights, with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 580  # 20s margin before 600s limit
    
    best_p = 0.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts):
        nonlocal best_p, best_support, best_weights
        if len(sup) != len(wts) or len(sup) == 0:
            return
        if np.any(np.isnan(sup)) or np.any(np.isinf(sup)):
            return
        if np.any(np.isnan(wts)) or np.any(np.isinf(wts)):
            return
        if np.any(sup < -1e-9):
            return
        if np.any(wts < -1e-9):
            return
        if wts.sum() <= 1e-9:
            return
        p_val = _compute_P(sup, wts)
        if p_val > best_p:
            best_p = p_val
            best_support = sup.copy()
            best_weights = wts.copy()
    
    # Start with baseline
    n_base = 32
    sup_base = np.arange(n_base, dtype=np.float64)
    w_base = np.ones(n_base, dtype=np.float64) / n_base
    update_best(sup_base, w_base)
    
    # Try Nelder-Mead optimization over (support, weights) for small n
    from scipy.optimize import minimize
    
    def objective(params):
        n = len(params) // 2
        s = params[:n]
        w = params[n:]
        # Support must be non-negative
        s = np.maximum(s, 0.0)
        w = np.maximum(w, 0.0)
        total_w = w.sum()
        if total_w <= 1e-9:
            return 0.0
        w_norm = w / total_w
        return -_compute_P(s, w_norm)
    
    def nelder_mead_trial(n_points, initial_s, initial_w, maxiter=200):
        """Run Nelder-Mead on concatenated (s, w) vector."""
        x0 = np.concatenate([initial_s, initial_w])
        res = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if res.success or res.fun < 0:
            x_opt = res.x
            s_opt = np.maximum(x_opt[:n_points], 0.0)
            w_opt = np.maximum(x_opt[n_points:], 0.0)
            if w_opt.sum() > 1e-9:
                w_opt = w_opt / w_opt.sum()
                update_best(s_opt, w_opt)
    
    # Try various n and random restarts
    rng = np.random.default_rng(42)
    
    # Different support grid patterns
    support_patterns = [
        # Uniform grid
        lambda n, scale: np.linspace(0, scale, n, dtype=np.float64),
        # Geometric spacing
        lambda n, scale: np.geomspace(0.01, scale, n, dtype=np.float64),
        # Mixed: some at 0, some spread out
        lambda n, scale: np.concatenate([[0.0] * max(1, n // 4), 
                                          np.linspace(1, scale, n - n // 4, dtype=np.float64)]),
        # Binary-like: mostly 0 and 1 with some intermediate
        lambda n, scale: np.sort(rng.uniform(0, scale, n).astype(np.float64)),
    ]
    
    scales = [4, 8, 16, 32]
    
    # Small n first (fast), then larger
    for n_points in [3, 4, 5, 6, 8, 12]:
        if time.time() > deadline - 30:
            break
        for scale in scales:
            if time.time() > deadline - 15:
                break
            for pat_idx, pattern in enumerate(support_patterns):
                if time.time() > deadline - 10:
                    break
                # Generate initial support
                s_init = pattern(n_points, scale).copy()
                # Generate initial weights
                w_init = rng.dirichlet(np.ones(n_points)).astype(np.float64)
                
                nelder_mead_trial(n_points, s_init, w_init, maxiter=300)
                
                if time.time() > deadline - 15:
                    break
    
    # Also try structured solutions: concentrate mass on few points
    # Try 2-point distributions
    for scale in [2, 4, 8, 16]:
        if time.time() > deadline - 10:
            break
        # Optimize over position a and probability p
        from scipy.optimize import minimize_scalar
        
        def two_point_obj(params):
            a, p = params
            a = max(0.0, a)
            p = np.clip(p, 0.0, 1.0)
            s = np.array([0.0, a], dtype=np.float64)
            w = np.array([1.0 - p, p], dtype=np.float64)
            return -_compute_P(s, w)
        
        best_two_p = 0.0
        best_two_params = None
        for p0 in np.linspace(0.1, 0.9, 9):
            for a0 in np.linspace(0.5, scale, 8):
                x0 = np.array([a0, p0])
                res = minimize(two_point_obj, x0, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-6})
                if res.fun < best_two_p:
                    best_two_p = res.fun
                    best_two_params = res.x
        
        if best_two_params is not None:
            a_opt = max(0.0, best_two_params[0])
            p_opt = np.clip(best_two_params[1], 0.0, 1.0)
            s_two = np.array([0.0, a_opt], dtype=np.float64)
            w_two = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
            update_best(s_two, w_two)
    
    # Try 3-point distributions with optimization
    for scale in [4, 8, 16]:
        if time.time() > deadline - 10:
            break
        def three_point_obj(params):
            a, b, p1, p2 = params
            a = max(0.0, a)
            b = max(0.0, b)
            # Ensure a < b
            if b <= a:
                b = a + 0.01
            p1 = np.clip(p1, 0.0, 1.0)
            p2 = np.clip(p2, 0.0, 1.0)
            total = p1 + p2
            if total > 1.0:
                p1 = p1 / total
                p2 = p2 / total
            p0 = max(0.0, 1.0 - p1 - p2)
            s = np.array([0.0, a, b], dtype=np.float64)
            w = np.array([p0, p1, p2], dtype=np.float64)
            if w.sum() <= 1e-9:
                return 0.0
            w = w / w.sum()
            return -_compute_P(s, w)
        
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            x0 = np.array([rng.uniform(0.5, scale * 0.5), 
                          rng.uniform(scale * 0.5, scale),
                          rng.uniform(0.2, 0.7),
                          rng.uniform(0.2, 0.7)])
            res = minimize(three_point_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6})
            if res.fun < 0:
                a, b, p1, p2 = res.x
                a = max(0.0, a)
                b = max(a + 0.001, b)
                p1 = np.clip(p1, 0.0, 1.0)
                p2 = np.clip(p2, 0.0, 1.0)
                total = p1 + p2
                if total > 1.0:
                    p1 = p1 / total
                    p2 = p2 / total
                p0 = max(0.0, 1.0 - p1 - p2)
                s_three = np.array([0.0, a, b], dtype=np.float64)
                w_three = np.array([p0, p1, p2], dtype=np.float64)
                if w_three.sum() > 1e-9:
                    w_three = w_three / w_three.sum()
                    update_best(s_three, w_three)
    
    # If we found something better than baseline, return it
    if best_support is not None and best_weights is not None:
        # Clean up: ensure non-negative
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        tw = best_weights.sum()
        if tw > 1e-9:
            best_weights = best_weights / tw
        return best_support.astype(np.float64), best_weights.astype(np.float64)
    
    # Fallback to baseline
    return sup_base, w_base


# EVOLVE-BLOCK-END
