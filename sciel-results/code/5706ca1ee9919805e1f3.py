import numpy as np
from scipy.optimize import minimize
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    # Pairwise sums S_ij = s_i + s_j with prob w_i * w_j
    S = support[:, None] + support[None, :]
    WS = weights[:, None] * weights[None, :]
    # For each (k, l), contribution is w_k * w_l * sum_{i,j: S_ij < 2*s_l - s_k} WS_ij
    # Vectorize: for each (k,l), threshold T = 2*s_l - s_k
    # T has shape (n, n) where T[k,l] = 2*s_l - s_k
    T = 2.0 * support[None, :] - support[:, None]  # T[k,l] = 2*s_l - s_k
    # For each (k,l), need sum of WS where S < T[k,l]
    # S has shape (n,n), T has shape (n,n) - but indices are different
    # S[i,j] is the sum, T[k,l] is the threshold
    # We need for each (k,l): sum_{i,j} WS[i,j] * (S[i,j] < T[k,l])
    # This is an n^4 computation. Let's use broadcasting carefully.
    # S_flat = S.ravel() shape (n^2,)
    # WS_flat = WS.ravel() shape (n^2,)
    # T_flat = T.ravel() shape (n^2,)
    # For each t in T_flat, sum WS_flat[S_flat < t]
    # Use searchsorted on sorted S
    S_flat = S.ravel()
    WS_flat = WS.ravel()
    # Sort S and accumulate WS
    idx = np.argsort(S_flat)
    S_sorted = S_flat[idx]
    WS_sorted = WS_flat[idx]
    WS_cumsum = np.cumsum(WS_sorted)
    T_flat = T.ravel()
    # For each T value, find how many S < T
    # np.searchsorted(S_sorted, T_flat, side='left') gives count of S < T
    counts = np.searchsorted(S_sorted, T_flat, side='left')
    # Sum of WS for those counts
    # WS_cumsum[counts-1] but handle counts=0
    WS_sums = np.zeros(len(T_flat))
    mask = counts > 0
    WS_sums[mask] = WS_cumsum[counts[mask] - 1]
    # Now P = sum_{k,l} w_k * w_l * WS_sums[k*n+l]
    WL = weights[:, None] * weights[None, :]  # WL[k,l] = w_k * w_l
    P = np.sum(WL * WS_sums.reshape(n, n))
    return P


def _evaluate_params(params, support_base):
    """Evaluate a parameter vector where params are weights (log-space or direct)."""
    weights = np.exp(params)
    return -_compute_P(support_base, weights)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    best_score = -1
    best_support = None
    best_weights = None
    
    # Try 2-point measures first: {0, a}
    # P = p * [3*p^2*(1-p) + (1-p)^3] where p = weight on 'a'
    # Optimize over p and a (a doesn't matter for 2-point as long as a > 0)
    for p in np.linspace(0.01, 0.99, 200):
        if time.time() > deadline:
            break
        support = np.array([0.0, 1.0])
        weights = np.array([1-p, p])
        P = _compute_P(support, weights)
        if P > best_score:
            best_score = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Try 3-point measures: {0, a, b} with a < b
    # Use grid search over ratios and weights
    # Let's fix b=1, vary a in (0,1), and weights
    a_vals = np.linspace(0.05, 0.95, 30)
    for a in a_vals:
        if time.time() > deadline:
            break
        support = np.array([0.0, a, 1.0])
        # Grid over weights: w0 + w1 + w2 = 1
        # Use simplex grid
        n_grid = 20
        for w2 in np.linspace(0.01, 0.98, n_grid):
            if time.time() > deadline:
                break
            for w1 in np.linspace(0.01, 0.98 - w2, n_grid):
                w0 = 1.0 - w1 - w2
                if w0 < 0.005:
                    continue
                weights = np.array([w0, w1, w2])
                P = _compute_P(support, weights)
                if P > best_score:
                    best_score = P
                    best_support = support.copy()
                    best_weights = weights.copy()
    
    # Try 4-point measures with optimization
    if time.time() < deadline - 100:
        # Start from best 3-point and add a 4th point
        # Use scipy optimization on a 4-point support
        for init_a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if time.time() > deadline - 60:
                break
            support_init = np.array([0.0, init_a, 0.7, 1.0])
            weights_init = np.array([0.3, 0.2, 0.2, 0.3])
            
            def objective(params):
                if time.time() > deadline - 30:
                    return 1e10
                # params: [log_w0, log_w1, log_w2, log_w3, s1, s2, s3] (s0=0 fixed)
                log_w = params[:4]
                s = params[4:]
                support = np.concatenate([[0.0], s])
                weights = np.exp(log_w)
                # Ensure support is sorted and non-negative
                if np.any(support < -1e-9) or np.any(weights < -1e-9):
                    return 1e10
                if np.sum(weights) < 1e-9:
                    return 1e10
                return -_compute_P(support, weights)
            
            try:
                result = minimize(objective, 
                                  np.concatenate([np.log(weights_init), support_init[1:]]),
                                  method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if time.time() > deadline:
                    break
                if result.fun < -best_score:
                    log_w = result.x[:4]
                    s = result.x[4:]
                    support = np.concatenate([[0.0], s])
                    weights = np.exp(log_w)
                    # Validate
                    if np.all(support >= -1e-9) and np.all(weights >= -1e-9) and np.sum(weights) > 1e-9:
                        P = -result.fun
                        if P > best_score:
                            best_score = P
                            best_support = support.copy()
                            best_weights = weights.copy()
            except Exception:
                pass
    
    # If we haven't found anything better, use the 2-point optimal
    if best_support is None:
        # Use the analytical 2-point optimum
        p_opt = 0.73
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([1 - p_opt, p_opt])
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    total_w = np.sum(best_weights)
    if total_w < 1e-9:
        best_support = np.array([0.0])
        best_weights = np.array([1.0])
    
    return best_support, best_weights
