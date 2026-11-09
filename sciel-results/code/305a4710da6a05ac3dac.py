# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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

def _optimize_4point():
    """Try a 4-point distribution {0, a, b, M}."""
    
    def prob_4point(params):
        p0, pa, pb, pm, a, b, M = params
        if any(p < 0 for p in [p0, pa, pb, pm]):
            return 0.0
        total_p = p0 + pa + pb + pm
        if total_p <= 0:
            return 0.0
        # Normalize
        p0, pa, pb, pm = p0/total_p, pa/total_p, pb/total_p, pm/total_p
        
        support = np.array([0.0, a, b, M])
        weights = np.array([p0, pa, pb, pm])
        return _prob_brute(support, weights)
    
    def neg_prob(params):
        return -prob_4point(params)
    
    # This is 7-dimensional, use differential evolution
    bounds = [
        (0, 1), (0, 1), (0, 1), (0, 1),  # weights
        (0.1, 5.0), (1.0, 20.0), (10.0, 500.0)  # support points
    ]
    
    try:
        result = differential_evolution(
            neg_prob, bounds,
            maxiter=50, popsize=10,
            tol=1e-8, seed=123,
            polish=True
        )
        best_params = result.x
        best_prob = -result.fun
    except Exception:
        # Fallback
        best_params = np.array([0.4, 0.2, 0.2, 0.2, 1.0, 5.0, 20.0])
        best_prob = prob_4point(best_params)
    
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
    
    # If we have time, try 4-point
    elapsed = time.time() - start_time
    if elapsed < 300:
        try:
            params_4, prob_4 = _optimize_4point()
            elapsed = time.time() - start_time
            
            if prob_4 > best_prob:
                best_prob = prob_4
                p0, pa, pb, pm, a, b, M = params_4
                total_p = p0 + pa + pb + pm
                if total_p > 0:
                    p0, pa, pb, pm = p0/total_p, pa/total_p, pb/total_p, pm/total_p
                    support_4 = np.array([0.0, a, b, M])
                    weights_4 = np.array([p0, pa, pb, pm])
                    weights_4 = np.maximum(weights_4, 0)
                    if weights_4.sum() > 1e-9:
                        weights_4 = weights_4 / weights_4.sum()
                        best_support = support_4
                        best_weights = weights_4
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

    return best_support, best_weights

# EVOLVE-BLOCK-END
