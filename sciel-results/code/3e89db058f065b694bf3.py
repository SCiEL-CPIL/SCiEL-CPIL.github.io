# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    w = weights / np.sum(weights)
    
    # Pairwise sums: s_i + s_j for all (i,j)
    pair_sums = support[:, None] + support[None, :]  # (n, n)
    
    # For each (k, l) pair: need sum of w_i * w_j where s_i + s_j < 2*s_l - s_k
    # P = sum_k sum_l w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    
    # Build matrix M[k,l] = sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j
    # Vectorize: for each l, compute threshold t_l = 2*s_l
    # For each k, threshold is t_l - s_k
    # We need cumulative sum of w_i*w_j over pair_sums < threshold
    
    # Sort pair_sums and compute cumulative weights
    flat_sums = pair_sums.ravel()  # (n*n,)
    flat_w = (w[:, None] * w[None, :]).ravel()  # (n*n,)
    
    # Sort by sum value
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_w = flat_w[order]
    cumsum_w = np.cumsum(sorted_w)
    
    # For each (k, l): threshold = 2*s_l - s_k
    # Find count of pair_sums < threshold using searchsorted
    # thresholds shape: (n, n) where thresholds[k,l] = 2*s_l - s_k
    thresholds = 2 * support[None, :] - support[:, None]  # (n, n): [k, l]
    
    # Use searchsorted to find number of elements < threshold
    # np.searchsorted returns index where threshold would be inserted
    # We want strict inequality: count of elements strictly less than threshold
    counts = np.searchsorted(sorted_sums, thresholds, side='left')  # (n, n)
    
    # Get cumulative sums at those indices (0 if count=0)
    # cumsum_w[i] = sum of first i elements
    # So sum of elements with index < count = cumsum_w[count] (with cumsum_w[0]=0)
    padded_cumsum = np.concatenate([[0.0], cumsum_w])
    inner = padded_cumsum[counts]  # (n, n): inner[k,l] = sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i*w_j
    
    # P = sum_k sum_l w_k * w_l * inner[k,l]
    P = np.sum(w[:, None] * w[None, :] * inner)
    return P


def _optimize_weights(support, max_time=500):
    """Optimize weights on fixed support to maximize P."""
    n = len(support)
    start_time = time.time()
    
    # Parameterize: use log-weights to ensure positivity
    # weights = exp(theta) / sum(exp(theta))
    best_theta = np.zeros(n)
    best_P = _compute_P(support, np.ones(n))
    
    def neg_P(theta):
        w = np.exp(theta - np.max(theta))
        w = w / np.sum(w)
        return -_compute_P(support, w)
    
    def grad_neg_P(theta):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros(n)
        base = neg_P(theta)
        for i in range(n):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            g[i] = (neg_P(theta_plus) - base) / eps
            theta_plus[i] -= 2 * eps
            g[i] = (neg_P(theta) - neg_P(theta_plus)) / eps
            theta_plus[i] += eps
        return g
    
    # Try multiple initializations
    initializations = [
        np.zeros(n),
        np.linspace(0, 1, n),
        np.linspace(1, 0, n),
        np.concatenate([np.ones(n//2), np.zeros(n - n//2)]),
        np.concatenate([np.zeros(n//2), np.ones(n - n//2)]),
    ]
    if n >= 4:
        mid = n // 2
        init_mid = np.zeros(n)
        init_mid[mid-1:mid+1] = 1.0
        initializations.append(init_mid)
    
    for init in initializations:
        if time.time() - start_time > max_time:
            break
        try:
            res = minimize(neg_P, init, method='L-BFGS-B', 
                          options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-8})
            if res.fun < -best_P:
                best_P = -res.fun
                best_theta = res.x
        except Exception:
            pass
    
    # Convert back to weights
    w = np.exp(best_theta - np.max(best_theta))
    w = w / np.sum(w)
    return w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Try structured supports (geometric sequences) and optimize weights on each.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s wall clock limit
    
    best_support = None
    best_weights = None
    best_P = -np.inf
    
    # Candidate supports: geometric sequences with various ratios
    # Also include uniform grid as baseline
    candidate_configs = []
    
    # Baseline: uniform on {0, 1, ..., n-1}
    for n in [16, 24, 32, 48, 64]:
        candidate_configs.append(('uniform', n, 1.0))
    
    # Geometric sequences: support = {0, r, r^2, ..., r^(n-1)}
    # Small r concentrates mass near 0, large r spreads it out
    for n in [8, 12, 16, 24, 32, 48]:
        for r in [0.1, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 4.0]:
            candidate_configs.append(('geo', n, r))
    
    # Power-of-2 supports: {0, 1, 2, 4, 8, ...}
    for n in [8, 12, 16, 20]:
        candidate_configs.append(('pow2', n, 2.0))
    
    # Power-of-3 supports
    for n in [8, 12, 16]:
        candidate_configs.append(('pow3', n, 3.0))
    
    # Mixed: some points near 0, some spread out
    for n in [16, 24, 32]:
        candidate_configs.append(('mixed', n, 0.5))
    
    for config_type, n, param in candidate_configs:
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break
        
        try:
            if config_type == 'uniform':
                support = np.arange(n, dtype=np.float64)
            elif config_type == 'geo':
                support = np.array([0.0] + [param**i for i in range(1, n)], dtype=np.float64)
                # Normalize to keep values reasonable
                if np.max(support) > 100:
                    support = support / np.max(support) * 10
            elif config_type == 'pow2':
                support = np.array([0.0] + [2.0**i for i in range(1, n)], dtype=np.float64)
                if np.max(support) > 1000:
                    support = support / np.max(support) * 100
            elif config_type == 'pow3':
                support = np.array([0.0] + [3.0**i for i in range(1, n)], dtype=np.float64)
                if np.max(support) > 10000:
                    support = support / np.max(support) * 100
            elif config_type == 'mixed':
                # Half points near 0 (uniform 0..5), half spread out
                half = n // 2
                near = np.arange(half, dtype=np.float64) * param
                far = 10 + np.arange(n - half, dtype=np.float64) * 5 * param
                support = np.concatenate([near, far])
            else:
                continue
            
            # Ensure support is non-negative and sorted
            support = np.sort(support)
            support = np.maximum(support, 0.0)
            
            # Quick evaluation with uniform weights
            w_uniform = np.ones(n, dtype=np.float64) / n
            P_uniform = _compute_P(support, w_uniform)
            
            if P_uniform > best_P:
                best_P = P_uniform
                best_support = support.copy()
                best_weights = w_uniform.copy()
            
            # Optimize weights if time permits
            if time.time() > deadline - 60:
                break
            
            w_opt, P_opt = _optimize_weights(support, max_time=deadline - time.time() - 60)
            
            if P_opt > best_P:
                best_P = P_opt
                best_support = support.copy()
                best_weights = w_opt.copy()
                
        except Exception:
            continue
    
    # Final safety check
    if best_support is None or best_weights is None:
        # Fallback to baseline
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64).ravel()
    best_weights = np.asarray(best_weights, dtype=np.float64).ravel()
    
    # Clean up any numerical issues
    best_support = np.where(np.isnan(best_support), 0.0, best_support)
    best_support = np.where(np.isinf(best_support), 0.0, best_support)
    best_support = np.maximum(best_support, 0.0)
    
    best_weights = np.where(np.isnan(best_weights), 0.0, best_weights)
    best_weights = np.where(np.isinf(best_weights), 0.0, best_weights)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights > 0
    if np.sum(best_weights) < 1e-9:
        best_weights = np.ones_like(best_weights)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
