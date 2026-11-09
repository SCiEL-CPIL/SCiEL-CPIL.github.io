# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _prob_3point(p, q, M):
    """Compute P[X1+X2+X3 < 2*X4] for 3-point distribution at {0, 1, M}.
    
    p = P(X=0), q = P(X=1), r = P(X=M) = 1-p-q
    """
    r = 1.0 - p - q
    if r < -1e-12:
        return 0.0
    r = max(r, 0.0)
    
    # X4 can be 0, 1, or M
    # If X4=0: need X1+X2+X3 < 0, impossible since all >= 0
    # If X4=1: need X1+X2+X3 < 2
    # If X4=M: need X1+X2+X3 < 2M
    
    # P(X1+X2+X3 < 2) for X in {0,1,M} with M >= 1
    # X1+X2+X3 < 2 means sum is 0 or 1
    # Sum=0: all three are 0: p^3
    # Sum=1: exactly one is 1, rest 0: 3*p^2*q
    # (M >= 1 so if any is M, sum >= M >= 1, and if M>1 sum >= M > 1 unless M=1)
    # Actually if M=1, then sum=1 can also happen with one 1 and rest 0, or... 
    # Let's handle M > 1 case first (which is what we want)
    
    if M > 1.0 + 1e-9:
        # Sum < 2: sum = 0 or 1
        # sum=0: p^3
        # sum=1: 3*p^2*q (one 1, two 0s)
        # (can't have M since M > 1)
        prob_x4_1 = p**3 + 3*p**2*q
        
        # P(X1+X2+X3 < 2M) for M > 1
        # Since M > 1, 2M > 2
        # X1+X2+X3 < 2M
        # Max possible sum is 3M
        # We need sum < 2M
        # Sum can be: 0, 1, M, M+1, 2M, 2M+1, 3M
        # Sum < 2M: 0, 1, M, M+1 (since 2M is not < 2M)
        
        # Let's compute P(sum = s) for each relevant s
        # Using generating function: (p + q*x + r*x^M)^3
        
        # Coefficient of x^0: p^3
        # Coefficient of x^1: 3*p^2*q
        # Coefficient of x^M: 3*p^2*r
        # Coefficient of x^(M+1): 3*p*q^2
        # Coefficient of x^(2M): 3*p*r^2
        # Coefficient of x^(2M+1): 3*q*r^2
        # Coefficient of x^(3M): r^3
        
        # P(sum < 2M) = P(sum=0) + P(sum=1) + P(sum=M) + P(sum=M+1)
        prob_x4_M = p**3 + 3*p**2*q + 3*p**2*r + 3*p*q**2
        
        # Total probability
        total = q * prob_x4_1 + r * prob_x4_M
        return total
    else:
        # M <= 1, handle generally with brute force
        support = np.array([0.0, 1.0, M])
        weights = np.array([p, q, r])
        return _prob_brute(support, weights)

def _prob_brute(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] by direct enumeration."""
    n = len(support)
    w = weights / weights.sum()
    
    # P = sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i,j) with s_i + s_j < 2*s_l - s_k)
    # Wait, let me re-read the metric:
    # P = Prob[X1 + X2 + X3 < 2*X4]
    # = sum_{x1,x2,x3,x4} P(X1=x1)P(X2=x2)P(X3=x3)P(X4=x4) * 1[x1+x2+x3 < 2*x4]
    
    # More efficiently:
    # P = sum_{x4} P(X4=x4) * P(X1+X2+X3 < 2*x4)
    # P(X1+X2+X3 < 2*x4) = sum_{x1,x2,x3: x1+x2+x3 < 2*x4} P(X1=x1)P(X2=x2)P(X3=x3)
    
    # For each x4, compute the probability that X1+X2+X3 < 2*x4
    # We can precompute the distribution of S = X1+X2+X3
    
    # Distribution of S = X1+X2+X3
    # S takes values in {s_i + s_j + s_k for all i,j,k}
    # But we can compute it via convolution
    
    # Simpler: for small n, just enumerate
    total_prob = 0.0
    for idx4 in range(n):
        w4 = w[idx4]
        threshold = 2.0 * support[idx4]
        # P(X1+X2+X3 < threshold)
        p_sum = 0.0
        for idx1 in range(n):
            for idx2 in range(n):
                for idx3 in range(n):
                    if support[idx1] + support[idx2] + support[idx3] < threshold:
                        p_sum += w[idx1] * w[idx2] * w[idx3]
        total_prob += w4 * p_sum
    
    return total_prob

def _optimize_3point():
    """Optimize the 3-point distribution {0, 1, M}."""
    
    def neg_prob(params):
        p, q, M = params
        if p < 0 or q < 0 or p + q > 1 or M < 0:
            return 1.0
        return -_prob_3point(p, q, M)
    
    # Initial guess
    x0 = [0.5, 0.3, 10.0]
    bounds = [(0, 1), (0, 1), (1.0, 1000.0)]
    
    # Use differential evolution for global search
    result = differential_evolution(
        neg_prob, bounds, 
        maxiter=200, popsize=15,
        tol=1e-10, seed=42,
        polish=True
    )
    
    best_p, best_q, best_M = result.x
    best_prob = -result.fun
    
    # Also try some manual refinements
    candidates = [
        (0.5, 0.3, 10.0),
        (0.6, 0.2, 20.0),
        (0.7, 0.1, 50.0),
        (0.4, 0.4, 5.0),
        (0.3, 0.5, 100.0),
        (0.8, 0.05, 1000.0),
        (0.2, 0.6, 10.0),
        (0.5, 0.5, 2.0),
        (0.1, 0.8, 50.0),
    ]
    
    for p, q, M in candidates:
        prob = _prob_3point(p, q, M)
        if prob > best_prob:
            best_prob = prob
            best_p, best_q, best_M = p, q, M
    
    # Fine-tune with local optimization
    if best_M > 1.0 + 1e-9:
        res_local = minimize(
            neg_prob, [best_p, best_q, best_M],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if res_local.fun < -best_prob:
            best_p, best_q, best_M = res_local.x
            best_prob = -res_local.fun
    
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
    
    result = differential_evolution(
        neg_prob, bounds,
        maxiter=100, popsize=10,
        tol=1e-8, seed=123,
        polish=True
    )
    
    best_params = result.x
    best_prob = -result.fun
    return best_params, best_prob

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a small-support distribution (3 or 4 points) optimized analytically.
    """
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes
    
    # Try 3-point optimization first (fast)
    try:
        p, q, M, prob_3 = _optimize_3point()
        elapsed = time.time() - start_time
        print(f"3-point optimization: p={p:.6f}, q={q:.6f}, M={M:.4f}, P={prob_3:.10f}, time={elapsed:.1f}s")
        
        support_3 = np.array([0.0, 1.0, M])
        weights_3 = np.array([p, q, 1.0 - p - q])
        # Ensure non-negative
        weights_3 = np.maximum(weights_3, 0)
        if weights_3.sum() > 1e-9:
            weights_3 = weights_3 / weights_3.sum()
    except Exception as e:
        print(f"3-point optimization failed: {e}")
        support_3 = np.array([0.0, 1.0, 10.0])
        weights_3 = np.array([0.5, 0.3, 0.2])
        prob_3 = _prob_brute(support_3, weights_3)
    
    best_support = support_3
    best_weights = weights_3
    best_prob = prob_3
    
    # If we have time, try 4-point
    elapsed = time.time() - start_time
    if elapsed < 300:  # More than 5 minutes left
        try:
            params_4, prob_4 = _optimize_4point()
            elapsed = time.time() - start_time
            print(f"4-point optimization: P={prob_4:.10f}, time={elapsed:.1f}s")
            
            if prob_4 > best_prob:
                p0, pa, pb, pm, a, b, M = params_4
                total_p = p0 + pa + pb + pm
                if total_p > 1e-9:
                    p0, pa, pb, pm = p0/total_p, pa/total_p, pb/total_p, pm/total_p
                    support_4 = np.array([0.0, a, b, M])
                    weights_4 = np.array([p0, pa, pb, pm])
                    weights_4 = np.maximum(weights_4, 0)
                    if weights_4.sum() > 1e-9:
                        weights_4 = weights_4 / weights_4.sum()
                        best_support = support_4
                        best_weights = weights_4
                        best_prob = prob_4
        except Exception as e:
            print(f"4-point optimization failed: {e}")
    
    # Final validation
    support = best_support.astype(np.float64)
    weights = best_weights.astype(np.float64)
    
    # Ensure validity
    if len(support) != len(weights) or len(support) < 1:
        # Fallback
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
    
    # Check for NaN, inf, negative values
    if np.any(np.isnan(support)) or np.any(np.isinf(support)):
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
    
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        weights = np.ones_like(support)
    
    if np.any(support < -1e-9):
        support = np.maximum(support, 0.0)
    
    if np.any(weights < -1e-9):
        weights = np.maximum(weights, 0.0)
    
    if weights.sum() <= 1e-9:
        weights = np.ones_like(support)
    
    return support, weights

# EVOLVE-BLOCK-END
