# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a small number of support points (n=4) and optimize both support
    positions and weights. Uses scale invariance (fix s0=0) and fast P computation
    via sorted pairwise sums with searchsorted.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 570  # 570s safety margin before 630s kill
    
    def compute_P_fast(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] using sorted pairwise sums for speed."""
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0
        w = weights / w_sum
        
        # Precompute sorted pairwise sums and their cumulative weights
        # pair_sums[i,j] = support[i] + support[j]
        pair_sums = np.add.outer(support, support)  # shape (n, n)
        pair_weights = np.outer(w, w)  # shape (n, n)
        
        # Flatten and sort by pair sum
        flat_sums = pair_sums.flatten()
        flat_weights = pair_weights.flatten()
        sort_idx = np.argsort(flat_sums, kind='mergesort')
        sorted_sums = flat_sums[sort_idx]
        sorted_cumw = np.cumsum(flat_weights[sort_idx])
        
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * support[l] - support[k]
                # Strict inequality: pair_sums < threshold
                idx = np.searchsorted(sorted_sums, threshold, side='left')
                inner = sorted_cumw[idx - 1] if idx > 0 else 0.0
                P += w[k] * w[l] * inner
        
        return P
    
    def neg_P(params):
        """Negative of P for minimization. params = [s1, s2, s3, w0, w1, w2, w3]
        Scale invariance: fix support[0] = 0, optimize s1, s2, s3 > 0."""
        s1, s2, s3 = params[0], params[1], params[2]
        w0, w1, w2, w3 = params[3], params[4], params[5], params[6]
        
        support = np.array([0.0, s1, s2, s3], dtype=np.float64)
        weights = np.array([w0, w1, w2, w3], dtype=np.float64)
        
        # Enforce non-negativity
        if s1 < 0 or s2 < 0 or s3 < 0:
            return 1e10
        if np.any(weights < -1e-12):
            return 1e10
        
        weights = np.maximum(weights, 0.0)
        if np.sum(weights) < 1e-12:
            return 1e10
        
        return -compute_P_fast(support, weights)
    
    # Initial guesses: various starting points [s1, s2, s3, w0, w1, w2, w3]
    inits = [
        # Uniform-like on {0, 1, 2, 3}
        np.array([1.0, 2.0, 3.0, 0.25, 0.25, 0.25, 0.25]),
        # Concentrated at 0
        np.array([5.0, 10.0, 20.0, 0.6, 0.2, 0.1, 0.1]),
        # Geometric-like
        np.array([1.0, 3.0, 9.0, 0.5, 0.3, 0.15, 0.05]),
        # Spread out
        np.array([2.0, 6.0, 18.0, 0.4, 0.3, 0.2, 0.1]),
        # Two-point-like (large K)
        np.array([1.0, 100.0, 200.0, 0.8, 0.1, 0.05, 0.05]),
        # Another spread
        np.array([0.5, 2.0, 8.0, 0.3, 0.3, 0.25, 0.15]),
        # Moderate spread
        np.array([1.0, 4.0, 16.0, 0.45, 0.3, 0.15, 0.1]),
        # Very large K
        np.array([2.0, 50.0, 500.0, 0.7, 0.2, 0.08, 0.02]),
    ]
    
    best_params = None
    best_P = -1.0
    
    bounds = [(0.001, None)] * 3 + [(0, 1)] * 4
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(
                neg_P,
                init,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.fun < best_P or best_params is None:
                params = res.x
                support = np.array([0.0, max(params[0], 0), max(params[1], 0), max(params[2], 0)])
                weights = np.maximum(params[3:], 0.0)
                
                if np.sum(weights) > 1e-12:
                    P_val = -res.fun
                    if P_val > best_P:
                        best_P = P_val
                        best_params = params.copy()
        except Exception as e:
            pass
    
    # Polish pass with Nelder-Mead for fine-tuning
    if best_params is not None and time.time() < deadline - 30:
        try:
            res_polish = minimize(
                neg_P,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            
            P_val = -res_polish.fun
            if P_val > best_P:
                best_P = P_val
                best_params = res_polish.x.copy()
        except Exception as e:
            pass
    
    # Final high-precision polish: use SLSQP with very tight tolerance from best point
    if best_params is not None and time.time() < deadline - 15:
        try:
            res_final = minimize(
                neg_P,
                best_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False}
            )
            
            P_val = -res_final.fun
            if P_val > best_P:
                best_P = P_val
                best_params = res_final.x.copy()
        except Exception as e:
            pass
    
    # If we found a good solution, use it
    if best_params is not None:
        support = np.array([0.0, max(best_params[0], 0.0), max(best_params[1], 0.0), max(best_params[2], 0.0)])
        weights = np.maximum(best_params[3:], 0.0)
        
        # Ensure validity
        if np.sum(weights) < 1e-9:
            weights = np.ones(4) / 4
            support = np.arange(4, dtype=np.float64)
        
        return support.astype(np.float64), weights.astype(np.float64)
    else:
        # Fallback to baseline
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights


# EVOLVE-BLOCK-END
