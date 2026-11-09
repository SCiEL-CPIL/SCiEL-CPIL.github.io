# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_P_fast(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] using sorted pair sums."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    
    # Precompute pair sums and probabilities
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    flat_sums = pair_sums.ravel()
    flat_probs = pair_probs.ravel()
    order = np.argsort(flat_sums)
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    cumsum_probs = np.cumsum(sorted_probs)
    
    # thresholds[k,l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]
    flat_thresholds = thresholds.ravel()
    
    counts = np.searchsorted(sorted_sums, flat_thresholds, side='left')
    sums_arr = np.zeros_like(flat_thresholds)
    mask = counts > 0
    sums_arr[mask] = cumsum_probs[counts[mask] - 1]
    
    sums_2d = sums_arr.reshape(n, n)
    weight_matrix = p[:, None] * p[None, :]
    P = np.sum(weight_matrix * sums_2d)
    return P


def _optimize_weights_fixed_support(support, initial_weights, deadline, max_iter=2000):
    """Optimize weights for a fixed support using coordinate ascent + scipy."""
    s = np.asarray(support, dtype=np.float64)
    n = len(s)
    w = np.asarray(initial_weights, dtype=np.float64).copy()
    total = w.sum()
    if total <= 1e-9:
        return w, 0.0
    w = w / total
    
    best_P = _compute_P_fast(s, w)
    best_w = w.copy()
    
    # Coordinate ascent on simplex
    step = 0.05
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try moving weight from i to j
            for j in range(n):
                if i == j:
                    continue
                for delta in [step, -step]:
                    new_w = w.copy()
                    new_w[i] -= delta
                    new_w[j] += delta
                    if new_w[i] < 1e-8 or new_w[j] < 1e-8:
                        continue
                    P = _compute_P_fast(s, new_w)
                    if P > best_P:
                        best_P = P
                        best_w = new_w.copy()
                        w = new_w.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
    
    # Try scipy optimization on log-weights
    if time.time() < deadline - 30:
        def neg_log_likelihood(log_w):
            w = np.exp(log_w)
            w = w / w.sum()
            P = _compute_P_fast(s, w)
            return -P
        
        # Start from current best
        w_init = best_w.copy()
        w_init = np.maximum(w_init, 1e-10)
        log_w_init = np.log(w_init / w_init.sum())
        
        try:
            result = minimize(neg_log_likelihood, log_w_init, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            if result.fun < -best_P:
                w_opt = np.exp(result.x)
                w_opt = w_opt / w_opt.sum()
                P_opt = _compute_P_fast(s, w_opt)
                if P_opt > best_P:
                    best_P = P_opt
                    best_w = w_opt.copy()
        except:
            pass
    
    return best_w, best_P


def _try_geometric_support(ratio, n_points, deadline):
    """Try a geometric sequence support and optimize weights."""
    if time.time() > deadline:
        return None, None, 0.0
    
    # Support: 0, 1, ratio, ratio^2, ..., ratio^(n-1)
    support = np.array([0.0] + [ratio**i for i in range(1, n_points)], dtype=np.float64)
    
    # Initial weights: decay
    weights = np.array([0.5 * (0.5 ** i) for i in range(n_points)], dtype=np.float64)
    weights = weights / weights.sum()
    
    best_w, best_P = _optimize_weights_fixed_support(support, weights, deadline)
    return support, best_w, best_P


def _try_two_point(M, p, deadline):
    """Try a two-point distribution."""
    support = np.array([0.0, M], dtype=np.float64)
    weights = np.array([p, 1-p], dtype=np.float64)
    P = _compute_P_fast(support, weights)
    return support, weights, P


def _try_three_point(a, b, p1, p2, deadline):
    """Try a three-point distribution at 0, a, b."""
    support = np.array([0.0, a, b], dtype=np.float64)
    weights = np.array([p1, p2, 1-p1-p2], dtype=np.float64)
    if weights.min() < 1e-8:
        return None, None, 0.0
    P = _compute_P_fast(support, weights)
    return support, weights, P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s
    
    best_P = -1
    best_support = None
    best_weights = None
    
    # Phase 1: Try two-point distributions
    for M in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        for p in np.linspace(0.05, 0.95, 50):
            if time.time() > deadline - 100:
                break
            support, weights, P = _try_two_point(M, p, deadline)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
    
    # Phase 2: Try geometric supports with optimized weights
    for ratio in [2.0, 3.0, 4.0, 5.0, 10.0]:
        for n_points in [3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 60:
                break
            support, weights, P = _try_geometric_support(ratio, n_points, deadline)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
    
    # Phase 3: Try structured supports with different patterns
    # Powers of 2
    for n_points in [3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 60:
            break
        support = np.array([2.0**i for i in range(n_points)], dtype=np.float64)
        # Try various initial weight patterns
        for decay in [0.3, 0.5, 0.7]:
            weights = np.array([decay**i for i in range(n_points)], dtype=np.float64)
            weights = weights / weights.sum()
            opt_w, P = _optimize_weights_fixed_support(support, weights, deadline)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = opt_w.copy()
    
    # Phase 4: Local refinement of best solution
    if best_support is not None and time.time() < deadline - 30:
        support = best_support.copy()
        weights = best_weights.copy()
        n = len(support)
        
        # Try adding a new point
        max_val = support.max()
        for new_pos in [0.0, max_val * 0.5, max_val * 2.0, max_val * 5.0]:
            if time.time() > deadline - 30:
                break
            new_support = np.append(support, new_pos)
            # Try small weights for new point
            for eps in [0.05, 0.1, 0.2, 0.3]:
                new_weights = np.append(weights * (1 - eps), eps)
                P = _compute_P_fast(new_support, new_weights)
                if P > best_P:
                    best_P = P
                    best_support = new_support.copy()
                    best_weights = new_weights.copy()
                    # Optimize further
                    opt_w, P2 = _optimize_weights_fixed_support(new_support, new_weights, deadline)
                    if P2 > best_P:
                        best_P = P2
                        best_weights = opt_w.copy()
        
        # Perturb support points
        step = 0.1 * max_val
        for _ in range(100):
            if time.time() > deadline - 15:
                break
            improved = False
            for i in range(len(best_support)):
                for delta in [-step, step]:
                    new_support = best_support.copy()
                    new_support[i] = max(0.0, best_support[i] + delta)
                    P = _compute_P_fast(new_support, best_weights)
                    if P > best_P:
                        best_P = P
                        best_support = new_support.copy()
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
    
    # Ensure validity
    if best_support is None or best_weights is None:
        n = 8
        support = np.array([2.0**i for i in range(n)], dtype=np.float64)
        weights = np.array([0.5**i for i in range(n)], dtype=np.float64)
        weights = weights / weights.sum()
        return support, weights
    
    # Final validation
    if len(best_support) != len(best_weights):
        n = 8
        support = np.array([2.0**i for i in range(n)], dtype=np.float64)
        weights = np.array([0.5**i for i in range(n)], dtype=np.float64)
        weights = weights / weights.sum()
        return support, weights
    
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9) or np.sum(best_weights) <= 1e-9:
        n = 8
        support = np.array([2.0**i for i in range(n)], dtype=np.float64)
        weights = np.array([0.5**i for i in range(n)], dtype=np.float64)
        weights = weights / weights.sum()
        return support, weights
    
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)) or \
       np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        n = 8
        support = np.array([2.0**i for i in range(n)], dtype=np.float64)
        weights = np.array([0.5**i for i in range(n)], dtype=np.float64)
        weights = weights / weights.sum()
        return support, weights
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
