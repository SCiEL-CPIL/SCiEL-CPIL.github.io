# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for i.i.d. X's from discrete measure.
    P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
    """
    n = len(support)
    s = support
    w = weights
    # Pairwise sums s_i + s_j
    pair_sums = s[:, None] + s[None, :]
    # Threshold for each (k, l): 2*s_l - s_k
    # shape (n, n) for (k, l)
    thresholds = 2.0 * s[None, :] - s[:, None]  # (k, l) -> 2*s_l - s_k
    # For each (k, l), count pairs (i,j) with pair_sums[i,j] < thresholds[k,l]
    # pair_sums shape (n, n), thresholds shape (n, n)
    # We need: for each (k,l), sum over (i,j) where pair_sums[i,j] < thresholds[k,l] of w_i*w_j
    # Build weight matrix for pairs
    w_pair = w[:, None] * w[None, :]  # (i, j) -> w_i * w_j
    # For each (k, l), inner = sum_{i,j: pair_sums[i,j] < thresholds[k,l]} w_pair[i,j]
    # This is O(n^4) if done naively. Use broadcasting with mask.
    # pair_sums[:, :, None, None] < thresholds[None, None, :, :]
    # This would be (n, n, n, n) which is too large for n > ~64.
    # Instead, loop over l or use sorting.
    
    # Alternative: for each l, precompute cumulative sums over pair_sums sorted.
    # But pair_sums depends on i,j jointly.
    
    # Let's use the fact that for fixed l, we need for each k:
    #   inner_k = sum_{i,j: s_i + s_j < 2*s_l - s_k} w_i * w_j
    # We can precompute for each possible threshold value the cumulative weight.
    # Since s_i + s_j takes values in {s_a + s_b : a, b in range(n)},
    # we can sort these and use cumulative sums.
    
    # Collect all pair sums and their weights
    all_pair_sums = pair_sums.ravel()  # (n*n,)
    all_pair_weights = w_pair.ravel()  # (n*n,)
    
    # Sort by pair sum
    sort_idx = np.argsort(all_pair_sums, kind='mergesort')
    sorted_sums = all_pair_sums[sort_idx]
    sorted_weights = all_pair_weights[sort_idx]
    cum_weights = np.cumsum(sorted_weights)
    
    # For each (k, l), threshold = 2*s_l - s_k
    # We need sum of weights where sorted_sums < threshold
    # Use searchsorted to find insertion point
    # thresholds shape (n, n) for (k, l)
    # searchsorted returns index where threshold would be inserted
    # We want strict inequality: s_i + s_j < threshold
    # So we want the number of elements strictly less than threshold
    insert_idx = np.searchsorted(sorted_sums, thresholds, side='left')  # (n, n)
    # cum_weights[insert_idx - 1] gives sum of weights for elements < threshold
    # Handle insert_idx = 0 case
    inner = np.zeros_like(thresholds)
    mask = insert_idx > 0
    inner[mask] = cum_weights[insert_idx[mask] - 1]
    
    # P = sum_{k,l} w_k * w_l * inner[k, l]
    w_kl = w[:, None] * w[None, :]  # (k, l)
    P = np.sum(w_kl * inner)
    return P


def _objective(params, n):
    """
    Objective function to minimize (negative P).
    params: [s_0, ..., s_{n-1}, w_0, ..., w_{n-1}]
    """
    s = params[:n]
    w = params[n:]
    
    # Enforce non-negativity via soft penalty
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return 1e6
    
    # Ensure support is sorted (helps with numerical stability)
    s_sorted_idx = np.argsort(s)
    s = s[s_sorted_idx]
    w = w[s_sorted_idx]
    
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 1e6
    w = w / w_sum
    
    P = _compute_P(s, w)
    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Uses gradient-based optimization over both support points and weights.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    # Start with a reasonable initial configuration
    # Use a small number of points to keep optimization tractable
    n = 12
    
    # Initial support: spread out points
    s_init = np.linspace(0, 10, n)
    w_init = np.ones(n) / n
    
    params_init = np.concatenate([s_init, w_init])
    
    # Bounds: support >= 0, weights >= 0
    bounds = [(0, 50)] * n + [(1e-6, 1.0)] * n
    
    best_params = params_init.copy()
    best_P = _compute_P(s_init, w_init)
    
    # Try multiple random restarts with different initializations
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            s_init = np.linspace(0, 10, n)
            w_init = np.ones(n) / n
        elif restart == 1:
            # Concentrated near 0
            s_init = np.concatenate([[0], np.linspace(0.1, 5, n-1)])
            w_init = np.ones(n) / n
        elif restart == 2:
            # Geometric spacing
            s_init = np.geomspace(1e-3, 20, n)
            w_init = np.ones(n) / n
        elif restart == 3:
            # Bimodal
            s_init = np.concatenate([np.linspace(0, 1, n//2), np.linspace(5, 10, n - n//2)])
            w_init = np.ones(n) / n
        else:
            # Random
            s_init = rng.uniform(0, 15, n)
            s_init = np.sort(s_init)
            w_init = rng.dirichlet(np.ones(n))
        
        params_init = np.concatenate([s_init, w_init])
        
        # Use L-BFGS-B for bounded optimization
        try:
            result = minimize(
                _objective,
                params_init,
                args=(n,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success and result.fun < -best_P:
                best_P = -result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead for non-smooth improvements
        if time.time() > deadline:
            break
        
        try:
            result2 = minimize(
                _objective,
                best_params,
                args=(n,),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result2.fun < -best_P:
                best_P = -result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Extract best solution
    s = best_params[:n]
    w = best_params[n:]
    
    # Clean up: ensure non-negativity and normalize
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 1e-10)
    w = w / np.sum(w)
    
    # Sort by support
    sort_idx = np.argsort(s)
    s = s[sort_idx]
    w = w[sort_idx]
    
    # Remove duplicate support points by merging weights
    unique_s, unique_indices = np.unique(s, return_index=True)
    if len(unique_s) < n:
        # Merge weights for duplicate support points
        new_w = np.zeros(len(unique_s))
        for i, idx in enumerate(unique_indices):
            mask = np.isclose(s, unique_s[i], atol=1e-10)
            new_w[i] = np.sum(w[mask])
        s = unique_s
        w = new_w
        w = w / np.sum(w)
    
    return s.astype(np.float64), w.astype(np.float64)


# EVOLVE-BLOCK-END
