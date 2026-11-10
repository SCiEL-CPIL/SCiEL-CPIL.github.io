# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.

    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, use 590s to be safe

    # Strategy: Use a small number of support points and optimize weights.
    # The key insight is that we want P[X1+X2+X3 < 2*X4] to be high.
    # This is equivalent to P[X1+X2+X3-2*X4 < 0].
    
    # Let's try a few different configurations of support points and optimize weights.
    # A good heuristic: put mass on 0 and some large numbers.
    # If we put mass on 0 and M (large), let's analyze:
    # X_i in {0, M}.
    # X1+X2+X3 < 2*X4
    # If X4=0, need X1+X2+X3 < 0 -> impossible (since Xi >= 0).
    # If X4=M, need X1+X2+3 < 2M.
    # Let p = P[X=M], q = P[X=0] = 1-p.
    # P[X4=M] = p.
    # Given X4=M, X1+X2+X3 can be:
    # 0 (prob q^3): 0 < 2M (True if M>0)
    # M (prob 3q^2p): M < 2M (True if M>0)
    # 2M (prob 3qp^2): 2M < 2M (False, strict inequality)
    # 3M (prob p^3): 3M < 2M (False)
    # So P[X1+X2+X3 < 2M | X4=M] = q^3 + 3q^2p = q^2(q+3p) = q^2(1+2p)
    # Total P = p * q^2 * (1+2p)
    # Let's maximize f(p) = p(1-p)^2(1+2p) for p in [0,1].
    # f(p) = p(1-2p+p^2)(1+2p) = p(1+2p-2p-4p^2+p^2+2p^3) = p(1-3p^2+2p^3) = p - 3p^3 + 2p^4
    # f'(p) = 1 - 9p^2 + 8p^3
    # f'(p) = (p-1)(8p^2 - p - 1) = (p-1)(4p+1)(2p-1)
    # Critical points at p=1, p=-1/4, p=1/2.
    # f(1/2) = 0.5 * 0.25 * 2 = 0.25.
    # f(0) = 0, f(1) = 0.
    # So max is 0.25 for two-point distribution {0, M} with p=0.5.
    
    # Can we do better with more points?
    # Uniform on {0, ..., n-1} gave score 1.0.
    # Score is proportional to P.
    # For n=32, P was some value.
    # If we can get P > 0.25, that's already better than the 2-point case.
    # Actually, the uniform distribution on {0, ..., n-1} likely has P > 0.25.
    # Let's check n=32 uniform.
    # But we can optimize weights for a fixed support.
    
    # Let's try optimizing weights for support {0, 1, ..., n-1} with n=10 or so.
    # Or maybe a non-uniform support.
    
    # Idea: Use a geometric-like distribution?
    # Or simply optimize weights for a small support set.
    
    # Let's define a function to compute P for given support and weights.
    def compute_P(support, weights):
        n = len(support)
        if n == 0:
            return 0.0
        # Normalize weights
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w = weights / w_sum
        
        # P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
        # This is O(n^4) if done naively.
        # We can precompute the sum of w_i w_j for all pairs (i,j).
        # Let S_{ij} = w_i w_j.
        # We need for each (k,l), sum of S_{ij} where s_i + s_j < 2*s_l - s_k.
        
        # Vectorized approach:
        # s_i + s_j < 2*s_l - s_k  <=> s_i + s_k < 2*s_l - s_j ? No.
        # s_i + s_j < 2*s_l - s_k
        # Let A = s_i + s_j. We need A < 2*s_l - s_k.
        
        # Precompute all pairwise sums s_i + s_j and their probabilities w_i w_j.
        # There are n^2 pairs.
        # Let's create arrays of length n^2.
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        pair_sums = support[i_idx] + support[j_idx]
        pair_weights = w[i_idx] * w[j_idx]
        
        # Sort pair_sums to use binary search?
        # For each (k,l), we need sum of pair_weights where pair_sums < 2*s_l - s_k.
        # This is a 2D loop over k,l (n^2 iterations), each requiring a sum over n^2 pairs.
        # Total O(n^4). For n=10, 10^4 = 10000. Fast.
        # For n=50, 50^4 = 6.25M. Still okay in Python if vectorized?
        # Let's try to vectorize the inner sum.
        
        # We can compute the condition for all (k,l) and (i,j) at once?
        # That would be n^4 array. For n=20, 20^4 = 160,000. Fine.
        # For n=50, 50^4 = 6,250,000. Fine.
        # For n=100, 100^4 = 100,000,000. Might be slow/memory heavy.
        
        # Let's use the n^4 vectorized approach for small n.
        # k, l are indices for X4 and "threshold".
        # i, j are indices for X1, X2.
        
        # We need: sum_{k,l,i,j} w_k w_l w_i w_j * I(s_i + s_j < 2*s_l - s_k)
        
        # Let's reshape.
        # w_k w_l is n^2.
        # w_i w_j is n^2.
        # Condition is n^4.
        
        # To save memory, we can loop over l (n iterations) and vectorize over k, i, j.
        # For fixed l:
        # Threshold T_{k} = 2*s_l - s_k.
        # We need sum_{k,i,j} w_k w_i w_j I(s_i+s_j < T_k).
        
        # Precompute pair_sums (n x n) and pair_weights (n x n).
        # Sort pair_sums?
        # For each k, we need sum of pair_weights where pair_sums < T_k.
        # This is a 1D problem for each k if we sort pair_sums.
        
        # Let's sort pair_sums.
        flat_sums = pair_sums.flatten()
        flat_weights = pair_weights.flatten()
        sort_idx = np.argsort(flat_sums)
        sorted_sums = flat_sums[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        cum_weights = np.cumsum(sorted_weights)
        
        # Now for each (k,l), find count/sum where sorted_sums < 2*s_l - s_k.
        # Use np.searchsorted.
        
        P = 0.0
        for l in range(n):
            target = 2.0 * support[l]
            # For all k, threshold is target - support[k].
            thresholds = target - support
            # Find insertion points in sorted_sums.
            # We want strict inequality: s_i + s_j < threshold.
            # np.searchsorted(sorted_sums, thresholds, side='left') gives index where threshold would be inserted to keep order.
            # Elements at indices < idx are < threshold.
            idxs = np.searchsorted(sorted_sums, thresholds, side='left')
            # Sum of weights for indices < idxs.
            # cum_weights[idxs-1] gives sum up to idxs-1.
            # Handle idxs=0.
            inner_sums = np.zeros(n)
            mask = idxs > 0
            inner_sums[mask] = cum_weights[idxs[mask] - 1]
            # P += sum_k w_k * w_l * inner_sums[k]
            P += w[l] * np.sum(w * inner_sums)
            
        return P

    # Objective function to minimize (negative of P)
    def objective(weights):
        # Ensure weights are non-negative and sum to 1 (or close)
        w = np.maximum(weights, 0.0)
        s = np.sum(w)
        if s < 1e-15:
            return 1.0  # Large penalty
        w = w / s
        return -compute_P(support_current, w)

    # Let's try a few support configurations.
    best_score = 0.0
    best_support = None
    best_weights = None

    # Config 1: Uniform support {0, ..., n-1}
    # Config 2: Optimized weights on {0, ..., n-1}
    # Config 3: Different supports?
    
    # Start with n=10, optimize weights.
    n_opt = 10
    support_current = np.arange(n_opt, dtype=np.float64)
    
    # Initial guess: uniform
    w0 = np.ones(n_opt) / n_opt
    
    # Use SLSQP
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_opt
    
    try:
        res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 100, 'ftol': 1e-8})
        if res.success:
            w_opt = np.maximum(res.x, 0.0)
            s = np.sum(w_opt)
            if s > 1e-9:
                w_opt = w_opt / s
                score = compute_P(support_current, w_opt)
                if score > best_score:
                    best_score = score
                    best_support = support_current.copy()
                    best_weights = w_opt.copy()
    except Exception:
        pass

    # Try different n
    for n_try in [5, 8, 12, 15, 20]:
        if time.time() > deadline - 10.0:
            break
        support_current = np.arange(n_try, dtype=np.float64)
        w0 = np.ones(n_try) / n_try
        bounds = [(0.0, 1.0)] * n_try
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 50, 'ftol': 1e-8})
            if res.success:
                w_opt = np.maximum(res.x, 0.0)
                s = np.sum(w_opt)
                if s > 1e-9:
                    w_opt = w_opt / s
                    score = compute_P(support_current, w_opt)
                    if score > best_score:
                        best_score = score
                        best_support = support_current.copy()
                        best_weights = w_opt.copy()
        except Exception:
            pass

    # Try a non-uniform support?
    # Maybe {0, 1, 2, ..., 10, 100}?
    # Or just stick with optimized weights on uniform grid.
    
    # If no better found, fall back to the original uniform n=32
    if best_score == 0.0 or best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Ensure validity
        if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
        elif np.sum(best_weights) <= 1e-9:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights


# EVOLVE-BLOCK-END
