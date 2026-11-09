# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.stats import qmc

def _prob_brute(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] by direct enumeration for small n."""
    n = len(support)
    w = weights / weights.sum()
    
    total_prob = 0.0
    for idx4 in range(n):
        w4 = w[idx4]
        threshold = 2.0 * support[idx4]
        p_sum = 0.0
        for idx1 in range(n):
            for idx2 in range(n):
                for idx3 in range(n):
                    if support[idx1] + support[idx2] + support[idx3] < threshold:
                        p_sum += w[idx1] * w[idx2] * w[idx3]
        total_prob += w4 * p_sum
    
    return total_prob

def _prob_3point_analytic(p, q, M):
    """
    Compute P[X1+X2+X3 < 2*X4] for 3-point distribution at {0, 1, M}.
    p = P(X=0), q = P(X=1), r = P(X=M) = 1-p-q
    """
    r = 1.0 - p - q
    if r < -1e-12:
        return 0.0
    r = max(r, 0.0)
    
    if M > 1.0 + 1e-9:
        # X4=0: P(sum<0) = 0
        # X4=1: P(sum<2) = p^3 + 3*p^2*q  (sums 0 or 1)
        prob_x4_1 = p**3 + 3*p**2*q
        
        # X4=M: P(sum<2M). For M>1, sums < 2M are: 0,1,2,M,M+1
        # sum=0: p^3
        # sum=1: 3*p^2*q
        # sum=2: 3*p*q^2  (two 1s, one 0)
        # sum=M: 3*p^2*r
        # sum=M+1: 3*p*q*r
        prob_x4_M = p**3 + 3*p**2*q + 3*p*q**2 + 3*p**2*r + 3*p*q*r
        
        total = q * prob_x4_1 + r * prob_x4_M
        return total
    else:
        # Fallback to brute force for M <= 1
        support = np.array([0.0, 1.0, M])
        weights = np.array([p, q, r])
        return _prob_brute(support, weights)

def _prob_5point_analytic(w0, w1, wa, wb, a, b):
    """
    Compute P[X1+X2+X3 < 2*X4] for 5-point distribution {0, 1, a, b, M} with M very large.
    w0+w1+wa+wb+wm = 1, where wm = 1-w0-w1-wa-wb
    For M very large, X4=M gives threshold 2M, so almost all sums < 2M.
    X4=0: threshold 0, P(sum<0)=0
    X4=1: threshold 2, need sum<2
    X4=a: threshold 2a, need sum<2a
    X4=b: threshold 2b, need sum<2b
    X4=M: threshold 2M, P(sum<2M)≈1 (all finite sums < 2M)
    """
    wm = 1.0 - w0 - w1 - wa - wb
    if wm < -1e-12:
        return 0.0
    wm = max(wm, 0.0)
    
    support = np.array([0.0, 1.0, a, b, 1e6])
    weights = np.array([w0, w1, wa, wb, wm])
    weights = np.maximum(weights, 0)
    if weights.sum() <= 1e-15:
        return 0.0
    weights = weights / weights.sum()
    
    # Use vectorized computation
    n = 5
    # Compute all pairwise sums s_i + s_j
    pair_sums = support[:, None] + support[None, :]  # (n, n)
    pair_probs = weights[:, None] * weights[None, :]  # (n, n)
    
    # For each x4, compute P(X1+X2+X3 < 2*x4)
    total_prob = 0.0
    for idx4 in range(n):
        threshold = 2.0 * support[idx4]
        # P(X1+X2+X3 < threshold) = sum over i,j,k with s_i+s_j+s_k < threshold of w_i*w_j*w_k
        # = sum over i,j of w_i*w_j * P(X3 < threshold - s_i - s_j)
        # For discrete: sum over i,j,k
        p_sum = 0.0
        for idx1 in range(n):
            for idx2 in range(n):
                rem = threshold - support[idx1] - support[idx2]
                # P(X3 < rem) = sum of w_k for s_k < rem
                mask = support < rem
                p_x3 = weights[mask].sum()
                p_sum += weights[idx1] * weights[idx2] * p_x3
        total_prob += weights[idx4] * p_sum
    
    return total_prob

def _optimize_3point():
    """Optimize the 3-point distribution {0, 1, M} using fine grid + local refinement."""
    
    best_p, best_q, best_M, best_prob = 0.5, 0.3, 10.0, 0.0
    
    # For M > 1, P = q*(p^3 + 3*p^2*q) + r*(p^3 + 3*p^2*q + 3*p*q^2 + 3*p^2*r + 3*p*q*r)
    # where r = 1-p-q
    # This is a function of (p, q, M) but for fixed p,q, P is linear in r (hence linear in (1-p-q))
    # Actually P depends on M only through the condition M>1. For M>1, P is independent of M!
    # So we just need to maximize over p, q with p+q < 1.
    
    # Let's verify: for M > 1, the formula doesn't depend on M.
    # So we maximize over (p, q) in the triangle p>=0, q>=0, p+q<=1.
    
    # Do a fine grid search over p and q
    n_grid = 200
    p_vals = np.linspace(0.0, 1.0, n_grid)
    q_vals = np.linspace(0.0, 1.0, n_grid)
    
    best_prob_grid = 0.0
    best_p_grid, best_q_grid = 0.5, 0.3
    
    for p in p_vals:
        for q in q_vals:
            if p + q >= 1.0:
                continue
            r = 1.0 - p - q
            if r < 0:
                continue
            # Use M=2 (any M>1 works)
            prob = _prob_3point_analytic(p, q, 2.0)
            if prob > best_prob_grid:
                best_prob_grid = prob
                best_p_grid, best_q_grid = p, q
    
    best_p, best_q, best_prob = best_p_grid, best_q_grid, best_prob_grid
    best_M = 2.0  # Any M > 1 gives same P
    
    # Refine with local optimization in (p, q) space
    def neg_prob_pq(params):
        p, q = params
        if p < 0 or q < 0 or p + q >= 1:
            return 1.0
        return -_prob_3point_analytic(p, q, 2.0)
    
    try:
        res_local = minimize(
            neg_prob_pq, [best_p, best_q],
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14}
        )
        if res_local.fun < -best_prob:
            best_p, best_q = res_local.x
            best_prob = -res_local.fun
    except Exception:
        pass
    
    # Try multiple starting points for local optimization
    start_points = [
        (0.3, 0.3), (0.4, 0.2), (0.5, 0.1), (0.2, 0.4),
        (0.6, 0.1), (0.1, 0.5), (0.5, 0.3), (0.3, 0.5),
    ]
    for sp in start_points:
        try:
            res = minimize(
                neg_prob_pq, sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res.fun < -best_prob:
                best_p, best_q = res.x
                best_prob = -res.fun
        except Exception:
            pass
    
    best_M = 2.0  # Any M > 1
    return best_p, best_q, best_M, best_prob

def _optimize_5point():
    """Try a 5-point distribution {0, 1, a, b, M} with M very large."""
    
    def prob_5point(params):
        w0, w1, wa, wb, a, b = params
        if w0 < -1e-12 or w1 < -1e-12 or wa < -1e-12 or wb < -1e-12:
            return 0.0
        w0 = max(w0, 0.0)
        w1 = max(w1, 0.0)
        wa = max(wa, 0.0)
        wb = max(wb, 0.0)
        wm = 1.0 - w0 - w1 - wa - wb
        if wm < -1e-12:
            return 0.0
        if a <= 0 or b <= a or a >= 1.0 or b >= 1.0:
            return 0.0
        return _prob_5point_analytic(w0, w1, wa, wb, a, b)
    
    def neg_prob(params):
        return -prob_5point(params)
    
    # 6-dimensional: w0, w1, wa, wb, a, b
    # Constraints: w0+w1+wa+wb <= 1, 0 < a < 1 < b
    bounds = [
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),  # weights
        (0.1, 0.99), (1.01, 50.0)  # a, b
    ]
    
    best_prob = 0.0
    best_params = None
    
    # Use multiple DE seeds for diversity
    for seed in [123, 456, 789]:
        try:
            result = differential_evolution(
                neg_prob, bounds,
                maxiter=30, popsize=15,
                tol=1e-6, seed=seed,
                polish=True
            )
            p = -result.fun
            if p > best_prob:
                best_prob = p
                best_params = result.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        best_params = np.array([0.3, 0.2, 0.1, 0.1, 0.5, 5.0])
        best_prob = prob_5point(best_params)
    
    return best_params, best_prob

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a small-support distribution (3 or 4 points) optimized analytically/numerically.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin
    
    best_support = None
    best_weights = None
    best_prob = -1.0
    
    # Try 3-point optimization first (fast and robust)
    try:
        p, q, M, prob_3 = _optimize_3point()
        elapsed = time.time() - start_time
        
        support_3 = np.array([0.0, 1.0, M])
        weights_3 = np.array([p, q, 1.0 - p - q])
        # Ensure non-negative
        weights_3 = np.maximum(weights_3, 0)
        if weights_3.sum() > 1e-9:
            weights_3 = weights_3 / weights_3.sum()
            
        if prob_3 > best_prob:
            best_prob = prob_3
            best_support = support_3
            best_weights = weights_3
    except Exception as e:
        pass
    
    # If we have time, try 5-point
    elapsed = time.time() - start_time
    if elapsed < 400:
        try:
            params_5, prob_5 = _optimize_5point()
            elapsed = time.time() - start_time
            
            if prob_5 > best_prob:
                best_prob = prob_5
                w0, w1, wa, wb, a, b = params_5
                w0 = max(w0, 0.0)
                w1 = max(w1, 0.0)
                wa = max(wa, 0.0)
                wb = max(wb, 0.0)
                wm = max(1.0 - w0 - w1 - wa - wb, 0.0)
                support_5 = np.array([0.0, 1.0, a, b, 1e6])
                weights_5 = np.array([w0, w1, wa, wb, wm])
                weights_5 = np.maximum(weights_5, 0)
                if weights_5.sum() > 1e-9:
                    weights_5 = weights_5 / weights_5.sum()
                    best_support = support_5
                    best_weights = weights_5
        except Exception as e:
            pass

    # Fallback if optimization failed
    if best_support is None:
        # Use the known good 3-point candidate
        best_support = np.array([0.0, 1.0, 1.5])
        best_weights = np.array([0.4, 0.1, 0.5])
        best_weights = best_weights / best_weights.sum()
    else:
        # Final validation
        if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
            best_support = np.array([0.0, 1.0, 1.5])
            best_weights = np.array([0.4, 0.1, 0.5])
            best_weights = best_weights / best_weights.sum()
        if np.sum(best_weights) <= 1e-9:
            best_support = np.array([0.0, 1.0, 1.5])
            best_weights = np.array([0.4, 0.1, 0.5])
            best_weights = best_weights / best_weights.sum()
        # Ensure support is sorted and valid
        best_support = np.sort(best_support)
        best_weights = np.maximum(best_weights, 0.0)
        best_weights = best_weights / best_weights.sum()

    return best_support, best_weights

# EVOLVE-BLOCK-END
