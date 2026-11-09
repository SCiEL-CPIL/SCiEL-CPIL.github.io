# EVOLVE-BLOCK-START
import numpy as np
import time

def calculate_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for discrete distribution.
    support: np.array of shape (n,)
    weights: np.array of shape (n,)
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    n = len(s)
    # Precompute pairwise sums s_i + s_j and their probabilities w_i * w_j
    # We need sum of w_i w_j for s_i + s_j < T
    # It is faster to iterate k, l and check i, j
    # n is small (<= 10), so O(n^4) is fine.
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum w_i * w_j for i, j such that s_i + s_j < threshold
            inner_sum = 0.0
            for i in range(n):
                for j in range(n):
                    if s[i] + s[j] < threshold:
                        inner_sum += w[i] * w[j]
            P += w[k] * w[l] * inner_sum
    return P

def optimize_measure(n, time_budget=60):
    """
    Optimize support and weights for a fixed n points.
    Supports are fixed to 0 = s[0] < s[1] < ... < s[n-1] = 1.
    Weights are non-negative and sum to 1.
    """
    if n < 2:
        return None, 0.0
    
    # Variables:
    # Supports: s[1] ... s[n-2] (since s[0]=0, s[n-1]=1)
    # Weights: w[0] ... w[n-2] (since sum w = 1, w[n-1] = 1 - sum(w[0..n-2]))
    
    num_support_vars = max(0, n - 2)
    num_weight_vars = n - 1
    
    total_vars = num_support_vars + num_weight_vars
    
    if total_vars == 0:
        # n=2 case, handled analytically or separately
        return None, 0.0

    def objective(params):
        # Unpack
        s_vars = params[:num_support_vars]
        w_vars = params[num_support_vars:]
        
        # Construct support
        s = np.zeros(n)
        s[0] = 0.0
        s[n-1] = 1.0
        if num_support_vars > 0:
            s[1:n-1] = s_vars
            
        # Ensure sorted (constraints in DE handle bounds, but let's be safe)
        # DE bounds ensure s_vars are in (0,1). We need s[1] < ... < s[n-2].
        # We can enforce this by sorting or using cumulative sums.
        # Cumulative sum approach is better for ordering.
        # Let's restart with a better parameterization for ordering.
        # Actually, DE with bounds on each var doesn't guarantee s[i] < s[i+1].
        # We should sort them. But if we sort, the mapping from params to s is non-smooth?
        # Sorting is differentiable almost everywhere. It should be okay for DE.
        if num_support_vars > 0:
            s[1:n-1] = np.sort(s_vars)
        
        # Construct weights
        w = np.zeros(n)
        w[:num_weight_vars] = w_vars
        w[n-1] = 1.0 - np.sum(w_vars)
        
        # Check validity
        if np.any(w < -1e-9):
            return 1.0 # Invalid, maximize by minimizing negative P? No, DE minimizes.
                       # Return large value to indicate bad.
        if np.any(w < 0):
             # Small negative due to floating point, clip to 0?
             # If significantly negative, return penalty.
             w = np.clip(w, 0, None)
             w_sum = np.sum(w)
             if w_sum == 0: return 1.0
             w = w / w_sum
             # Re-eval with clipped weights? Or just return penalty.
             # Let's return a penalty.
             pass 
        # For DE, we want to MINIMIZE -P.
        # So return -calculate_P(s, w)
        
        # Validity check for support
        if np.any(s < 0) or np.any(s > 1):
            return 1.0
            
        p_val = calculate_P(s, w)
        return -p_val

    # Bounds
    bounds = []
    # Support vars (internal): must be in (0, 1)
    for _ in range(num_support_vars):
        bounds.append((1e-6, 1.0 - 1e-6))
    # Weight vars: must be in (0, 1)
    for _ in range(num_weight_vars):
        bounds.append((1e-6, 1.0 - 1e-6))
        
    # Note: The constraint sum(w_vars) < 1 is not enforced by bounds.
    # We need to handle the case where sum(w_vars) >= 1.
    # If sum(w_vars) >= 1, then w[n-1] <= 0.
    # We can add a penalty in the objective if w[n-1] < 0.
    
    def penalized_objective(params):
        s_vars = params[:num_support_vars]
        w_vars = params[num_support_vars:]
        
        w_sum_first = np.sum(w_vars)
        if w_sum_first >= 1.0:
            # Penalty proportional to violation
            return 1.0 + (w_sum_first - 1.0)
            
        return objective(params)

    try:
        from scipy.optimize import differential_evolution
        # Use differential evolution
        # maxiter and popsize to fit within time_budget
        # Rough estimate: 1000 evals per second? 
        # Let's cap function evaluations.
        max_evals = 2000
        result = differential_evolution(
            penalized_objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-6,
            polish=True
        )
        
        if result.success or result.fun < 1.0:
            best_params = result.x
            s_vars = best_params[:num_support_vars]
            w_vars = best_params[num_support_vars:]
            
            s = np.zeros(n)
            s[0] = 0.0
            s[n-1] = 1.0
            if num_support_vars > 0:
                s[1:n-1] = np.sort(s_vars)
                
            w = np.zeros(n)
            w[:num_weight_vars] = w_vars
            w[n-1] = 1.0 - np.sum(w_vars)
            
            if np.any(w < -1e-9):
                 return None, 0.0, 0.0
                 
            w = np.clip(w, 0, None)
            w_sum = np.sum(w)
            if w_sum == 0: return None, 0.0, 0.0
            w = w / w_sum
            
            p_val = calculate_P(s, w)
            return s, w, p_val
        else:
            return None, 0.0, 0.0
    except Exception as e:
        return None, 0.0, 0.0

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = 550 # 50s margin
    
    best_support = None
    best_weights = None
    best_p = 0.0
    
    # 1. Analytic n=2 solution
    # From inspiration 1: p_opt = (15 - sqrt(33)) / 16.0
    # Wait, let's re-verify the analytic solution for n=2.
    # Let support be {0, 1}. Weights p, 1-p.
    # P[X1+X2+X3 < 2X4]
    # X4 can be 0 or 1.
    # If X4=0 (prob 1-p): 2X4=0. X1+X2+X3 < 0 is impossible since Xi>=0. Prob 0.
    # If X4=1 (prob p): 2X4=2. X1+X2+X3 < 2.
    # X1,X2,X3 are 0 or 1. Sum < 2 means Sum = 0 or 1.
    # P(Sum=0) = (1-p)^3
    # P(Sum=1) = 3 * p * (1-p)^2
    # Total P = p * [ (1-p)^3 + 3p(1-p)^2 ]
    # Let f(p) = p * (1-p)^2 * (1-p + 3p) = p * (1-p)^2 * (1+2p)
    # f(p) = p(1 - 2p + p^2)(1+2p) = p(1 + 2p - 2p - 4p^2 + p^2 + 2p^3) = p(1 - 3p^2 + 2p^3) = p - 3p^3 + 2p^4
    # f'(p) = 1 - 9p^2 + 8p^3
    # Set to 0: 8p^3 - 9p^2 + 1 = 0
    # (p-1) is a root? 8-9+1=0. Yes.
    # (p-1)(8p^2 - p - 1) = 0
    # Roots of 8p^2 - p - 1 = 0: p = (1 +/- sqrt(1 + 32)) / 16 = (1 +/- sqrt(33)) / 16
    # Positive root: (1 + sqrt(33)) / 16.
    # So p = (1 + sqrt(33)) / 16.
    # Inspiration 1 used (15 - sqrt(33))/16 which is 1 - (1+sqrt(33))/16. This is the weight at 0.
    # So weight at 1 is p = (1+sqrt(33))/16.
    
    p_at_1 = (1.0 + np.sqrt(33.0)) / 16.0
    p_at_0 = 1.0 - p_at_1
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_at_0, p_at_1])
    p2 = calculate_P(s2, w2)
    
    if p2 > best_p:
        best_p = p2
        best_support = s2.copy()
        best_weights = w2.copy()
        
    if time.time() > deadline:
        return best_support, best_weights

    # 2. Optimize n=3
    res3 = optimize_measure(3, time_budget=50)
    if res3 is not None and res3[2] > best_p:
        best_p = res3[2]
        best_support = res3[0].copy()
        best_weights = res3[1].copy()
        
    if time.time() > deadline:
        return best_support, best_weights

    # 3. Optimize n=4
    res4 = optimize_measure(4, time_budget=100)
    if res4 is not None and res4[2] > best_p:
        best_p = res4[2]
        best_support = res4[0].copy()
        best_weights = res4[1].copy()

    if time.time() > deadline:
        return best_support, best_weights

    # 4. Optimize n=5
    res5 = optimize_measure(5, time_budget=150)
    if res5 is not None and res5[2] > best_p:
        best_p = res5[2]
        best_support = res5[0].copy()
        best_weights = res5[1].copy()

    # Final cleanup
    if best_support is None:
        # Fallback to uniform
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)/n

    # Ensure types
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
