import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Co-optimize both support points and weights for a 3-point measure.
    We parameterize the support as [0, a, b] with 0 < a < b.
    We parameterize the weights as [p0, pa, pb] with p0+pa+pb=1.
    We use scipy.optimize.minimize (Nelder-Mead) to find the global maximum
    of the probability function over this 5D space (a, b, p0, pa).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    def calc_prob_params(params):
        """
        params: [a, b, p0, pa]
        Support: [0, a, b]
        Weights: [p0, pa, pb] where pb = 1 - p0 - pa
        """
        a, b, p0, pa = params
        
        # Basic validity checks for optimization
        if a <= 0 or b <= a or p0 < 0 or pa < 0:
            return -np.inf
        pb = 1.0 - p0 - pa
        if pb < 0:
            return -np.inf
        
        support = np.array([0.0, a, b])
        weights = np.array([p0, pa, pb])
        
        # Compute P[X1+X2+X3 < 2*X4]
        # X4 can be 0, a, or b.
        
        # Case X4 = 0 (prob p0):
        # 2*X4 = 0. X1+X2+X3 < 0 is impossible since support >= 0.
        # Contribution: 0
        
        # Case X4 = a (prob pa):
        # 2*X4 = 2a.
        # We need sum of 3 items from {0, a, b} to be < 2a.
        # Possible sums:
        # 0+0+0 = 0 < 2a (True if a>0) -> prob p0^3
        # 0+0+a = a < 2a (True if a>0) -> prob 3*p0^2*pa
        # 0+0+b = b < 2a? Depends on b vs 2a.
        # 0+a+a = 2a < 2a? False (strict inequality).
        # 0+a+b = a+b < 2a? b < a? False since b > a.
        # 0+b+b = b+b < 2a? 2b < 2a? False.
        # a+a+a = 3a < 2a? False.
        # ... any with b is >= b > a, so likely false unless b is small, but b>a.
        # So only 0+0+0 and permutations of 0+0+a matter.
        # Wait, is it possible that b < 2a?
        # If b < 2a, then 0+0+b is also a win.
        
        prob_x4_a = 0.0
        # 0+0+0
        prob_x4_a += p0**3
        # 0+0+a (3 perms)
        prob_x4_a += 3 * p0**2 * pa
        
        # Check if 0+0+b is a win
        if b < 2*a:
            prob_x4_a += 3 * p0**2 * pb
            
        # Check if 0+a+? 
        # 0+a+a = 2a (No)
        # 0+a+b > a+a = 2a (No)
        
        # Any sum with 2 or 3 'a's or 'b's will be >= 2a.
        # So contribution from X4=a is pa * prob_x4_a
        
        # Case X4 = b (prob pb):
        # 2*X4 = 2b.
        # We need sum of 3 items from {0, a, b} to be < 2b.
        # Total prob is 1. We can compute P[sum >= 2b] and subtract.
        # Sums >= 2b:
        # b+b+b = 3b >= 2b (Yes) -> prob pb^3
        # b+b+a = 2b+a >= 2b (Yes) -> prob 3*pb^2*pa
        # b+b+0 = 2b >= 2b (Yes, strict inequality fails) -> prob 3*pb^2*p0
        # b+a+a = b+2a. Is b+2a >= 2b? i.e., 2a >= b?
        # If 2a >= b, then b+a+a >= 2b.
        # If 2a < b, then b+a+a < 2b.
        
        prob_sum_ge_2b = 0.0
        prob_sum_ge_2b += pb**3
        prob_sum_ge_2b += 3 * pb**2 * pa
        prob_sum_ge_2b += 3 * pb**2 * p0
        
        if 2*a >= b:
            prob_sum_ge_2b += 3 * pb * pa**2
            # b+a+b is counted in 3*pb^2*pa? No.
            # Let's list all perms of (b,a,a):
            # (b,a,a), (a,b,a), (a,a,b) -> 3 perms. Prob 3*pb*pa^2.
            # Are there others?
            # (b,b,0) handled.
            # (b,b,a) handled.
            # (b,b,b) handled.
            # (a,a,a) = 3a. If 3a >= 2b?
            # If 2a >= b, then 3a > 2a >= b. Is 3a >= 2b?
            # If a=1, b=2, 3a=3, 2b=4. 3<4.
            # If a=1.5, b=2, 3a=4.5, 2b=4. 4.5>=4.
            # So we need to check 3a >= 2b as well.
            
            # Let's just iterate through all 27 combinations for X4=b case to be safe and simple,
            # since n=3 is small. It's O(1) per evaluation.
            
            # Actually, let's just compute the sum directly for all 27 cases.
            # It's robust.
            
            pass # We will replace the analytical derivation with a direct sum for correctness in the optimizer.
            
        # Let's rewrite the whole probability calculation using the direct sum method
        # which is O(n^3) but n=3, so 27 operations. Very fast.
        
        s = support
        w = weights
        
        p_total = 0.0
        for k in range(3): # X4 index
            threshold = 2.0 * s[k]
            prob_x4 = w[k]
            
            prob_sum_lt = 0.0
            for i in range(3): # X1
                for j in range(3): # X2
                    for l in range(3): # X3
                        if s[i] + s[j] + s[l] < threshold:
                            prob_sum_lt += w[i] * w[j] * w[l]
            
            p_total += prob_x4 * prob_sum_lt
            
        return p_total

    # Objective function to minimize (negative of probability)
    def objective(params):
        p = calc_prob_params(params)
        if np.isnan(p) or np.isinf(p):
            return 100.0
        return -p

    # Initial guesses
    # We want to explore the space where a and b are distinct.
    # The previous best was around support [0, 1, 10].
    # Let's try a few different starting points to avoid local minima.
    
    best_score = -np.inf
    best_params = None
    
    # Starting points: (a, b, p0, pa)
    # p0, pa must be positive, pb = 1-p0-pa > 0.
    starts = [
        (1.0, 10.0, 0.5, 0.3),  # Near previous best structure
        (1.0, 5.0, 0.4, 0.4),
        (0.5, 2.0, 0.5, 0.3),
        (2.0, 8.0, 0.6, 0.2),
        (0.1, 1.0, 0.7, 0.2),
        (1.0, 1.5, 0.5, 0.4), # Close supports
    ]
    
    # Constraints for Nelder-Mead are handled by bounds or penalty, 
    # but Nelder-Mead doesn't support bounds directly in older scipy versions 
    # or requires careful initialization. 
    # We'll use a penalty approach in the objective if bounds are violated, 
    # but we initialized with valid points.
    # Actually, let's use L-BFGS-B which supports bounds, or just Nelder-Mead 
    # and hope it stays in the valid region, or add a strong penalty.
    
    # Let's add a penalty to objective if params are invalid
    def penalized_objective(params):
        a, b, p0, pa = params
        # Penalty for invalid support
        if a <= 0 or b <= a:
            return 100.0 + abs(a) + abs(b)
        # Penalty for invalid weights
        if p0 < 0 or pa < 0 or p0 + pa > 1.0:
            return 100.0 + abs(p0) + abs(pa)
        
        return objective(params)

    for start in starts:
        if time.time() > deadline - 20:
            break
            
        try:
            res = minimize(penalized_objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_score:
                # Verify the result is actually valid
                p_val = calc_prob_params(res.x)
                if p_val > best_score and np.isfinite(p_val):
                    best_score = p_val
                    best_params = res.x
        except Exception:
            continue

    # If optimization failed or didn't find anything better than a baseline,
    # fall back to a known good configuration from the previous strategy
    if best_params is None or best_score < 0.4:
        # Use the previous best logic: support [0, 1, 10], weights optimized roughly
        # From previous code, best was around p0=0.5, p1=0.3, pM=0.2?
        # Let's just use a safe static value that scored 1.1
        # The previous program found a specific max. Let's re-run the grid search 
        # for the fixed support [0,1,10] quickly to ensure we have a baseline.
        
        # Actually, the previous program's score was 1.1018.
        # Let's just use a simple valid distribution that is likely decent.
        # Uniform on [0, 1, 10] gives P = ?
        # But we can do better.
        
        # Let's try one more specific optimization for fixed support [0, 1, 10]
        # to recover the previous score if the co-optimization failed.
        
        def calc_prob_fixed_01M(p0, p1):
            pM = 1 - p0 - p1
            if p0 <= 0 or p1 <= 0 or pM <= 0:
                return -np.inf
            # Same logic as before
            # X4=0: 0
            # X4=1: S < 2. S=0 (p0^3), S=1 (3p0^2p1). S=10 no.
            term1 = p1 * (p0**3 + 3*p0**2*p1)
            # X4=10: S < 20.
            # S >= 20:
            # 10+10+10=30 (pM^3)
            # 10+10+1=21 (3pM^2p1)
            # 10+10+0=20 (3pM^2p0) -- Strictly < 20, so 20 is excluded.
            term2 = pM * (1 - (pM**3 + 3*pM**2*p1 + 3*pM**2*p0))
            return term1 + term2

        # Grid search for fixed support
        best_fixed_score = -np.inf
        best_fixed_p0 = 0.5
        best_fixed_p1 = 0.3
        for p0 in np.arange(0.01, 0.98, 0.01):
            for p1 in np.arange(0.01, 0.98 - p0, 0.01):
                pM = 1 - p0 - p1
                if pM <= 0: continue
                p_val = calc_prob_fixed_01M(p0, p1)
                if p_val > best_fixed_score:
                    best_fixed_score = p_val
                    best_fixed_p0 = p0
                    best_fixed_p1 = p1
        
        if best_fixed_score > best_score:
            best_score = best_fixed_score
            # Construct params for fallback? No, just return the arrays directly.
            support = np.array([0.0, 1.0, 10.0])
            weights = np.array([best_fixed_p0, best_fixed_p1, 1.0 - best_fixed_p0 - best_fixed_p1])
            return support, weights

    # Extract best parameters
    a, b, p0, pa = best_params
    pb = 1.0 - p0 - pa
    
    # Ensure strict ordering and positivity
    if a <= 0: a = 1e-6
    if b <= a: b = a + 1e-6
    if p0 < 0: p0 = 1e-6
    if pa < 0: pa = 1e-6
    if pb < 0: pb = 1e-6
    
    # Normalize weights to be safe
    total_w = p0 + pa + pb
    p0 /= total_w
    pa /= total_w
    pb /= total_w
    
    support = np.array([0.0, a, b], dtype=np.float64)
    weights = np.array([p0, pa, pb], dtype=np.float64)
    
    # Final validity check
    if np.any(weights < -1e-9) or np.sum(weights) <= 1e-9 or np.any(support < -1e-9):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    return support, weights

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(s, w)
