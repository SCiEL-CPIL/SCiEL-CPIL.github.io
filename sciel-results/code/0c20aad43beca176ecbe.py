import numpy as np
from scipy.optimize import differential_evolution


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average probability measure.
    
    We design a discrete measure mu on [0, ∞) to maximize
    P(X1 + X2 + X3 < 2*X4) where X1..X4 are i.i.d. from mu.
    
    Approach: Use a small number of support points with optimized weights.
    We use differential evolution to search over support points and weights.
    The problem is scale-invariant, so we normalize the support points by their max
    inside the objective function to improve optimization stability.
    The objective function is fully vectorized for speed.
    """
    def neg_probability(params):
        n = len(params) // 2
        support = params[:n]
        weights = params[n:]
        
        # Ensure valid values
        if np.any(support < -1e-9) or np.any(weights < -1e-9):
            return 1.0  # Large penalty
        
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 1.0
        
        # Normalize weights
        w = weights / w_sum
        
        # Scale invariance: normalize support by its max
        s_max = np.max(support)
        if s_max <= 1e-9:
            return 1.0
        s = support / s_max
        
        # Compute all pairwise sums and weights
        s_i = s[:, np.newaxis]
        s_j = s[np.newaxis, :]
        sums_ij = (s_i + s_j).flatten()
        
        w_i = w[:, np.newaxis]
        w_j = w[np.newaxis, :]
        weights_ij = (w_i * w_j).flatten()
        
        # Sort sums_ij and permute weights_ij accordingly
        sort_idx = np.argsort(sums_ij)
        sums_sorted = sums_ij[sort_idx]
        weights_sorted = weights_ij[sort_idx]
        
        # Cumulative sum of weights
        cum_weights = np.cumsum(weights_sorted)
        
        # Compute thresholds T[k,l] = 2*s_l - s_k
        T = 2.0 * s[np.newaxis, :] - s[:, np.newaxis]
        T_flat = T.flatten()
        
        # For each threshold, find how many sums are strictly less
        idxs = np.searchsorted(sums_sorted, T_flat, side='right')
        
        # Get cumulative sum at those indices
        inner_sums = np.zeros(len(T_flat))
        mask = idxs > 0
        inner_sums[mask] = cum_weights[idxs[mask] - 1]
        
        inner_sums = inner_sums.reshape(n, n)
        
        # P = sum_{k,l} w_k w_l inner_sums[k,l]
        P = np.sum(w[:, np.newaxis] * w[np.newaxis, :] * inner_sums)
        
        return -P

    best_support = None
    best_weights = None
    best_score = -1.0

    # Try n=3, 4, 5
    for n in [3, 4, 5]:
        bounds = [(0.0, 10.0)] * n + [(0.0, 1.0)] * n
        
        result = differential_evolution(
            neg_probability,
            bounds=bounds,
            seed=42,
            maxiter=1500,
            tol=1e-10,
            polish=True,
            workers=1,
            popsize=15
        )
        
        if np.isfinite(result.fun):
            score = -result.fun
            if score > best_score:
                support = result.x[:n].copy()
                weights = result.x[n:].copy()
                
                # Ensure non-negative and valid
                support = np.maximum(support, 0.0)
                weights = np.maximum(weights, 1e-10)
                
                w_sum = np.sum(weights)
                if w_sum <= 1e-9:
                    continue
                
                weights = weights / w_sum
                
                # Normalize support to max 1 for consistency
                s_max = np.max(support)
                if s_max > 1e-9:
                    support = support / s_max
                else:
                    continue
                    
                best_score = score
                best_support = support
                best_weights = weights

    # Analytical 2-point solution score
    p_opt = (1 + np.sqrt(33)) / 16.0
    analytical_score = p_opt * (1 - p_opt)**2 * (1 + 2 * p_opt)
    
    if best_support is None or best_score < analytical_score:
        support = np.array([0.0, 1.0])
        weights = np.array([1-p_opt, p_opt])
        return support.astype(np.float64), weights.astype(np.float64)

    return best_support.astype(np.float64), best_weights.astype(np.float64)
