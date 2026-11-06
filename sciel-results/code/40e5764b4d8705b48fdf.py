import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    target = Fraction(5, 31)
    
    # Precompute harmonic sums up to a reasonable limit
    # We expect K to be small, maybe < 10000
    MAX_K = 20000
    H = [0.0] * (MAX_K + 1)
    for i in range(1, MAX_K + 1):
        H[i] = H[i-1] + 1.0/i
        
    def can_solve(r, last_d, K):
        """
        Returns a list of denominators if r can be represented by distinct 
        denominators in (last_d, K], else None.
        """
        if r == 0:
            return []
        if r < 0:
            return None
            
        # d must be > last_d and 1/d <= r => d >= ceil(1/r)
        # 1/r = r.denominator / r.numerator
        d_min = (r.denominator + r.numerator - 1) // r.numerator
        if d_min <= last_d:
            d_min = last_d + 1
            
        if d_min > K:
            return None
            
        # Pruning: Max possible sum with denominators d_min..K
        # Sum = H[K] - H[d_min-1]
        # If r > this sum, impossible.
        # Note: H is float, r is Fraction. Convert r to float for comparison?
        # Risk of precision loss. 
        # Better: compare r.numerator * (H[K] - H[d_min-1]) > r.denominator ?
        # Or just use Fraction for the sum? Too slow.
        # Let's use float but with a small epsilon.
        max_sum = H[K] - H[d_min-1]
        if float(r) > max_sum + 1e-9:
            return None
            
        # Iterate possible d
        # To find the solution with smallest max denominator, we are fixing K in binary search.
        # So any solution is valid.
        for d in range(d_min, K + 1):
            # 1/d <= r is guaranteed by d >= d_min
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break # Since d is increasing, 1/d is decreasing, so new_r will become more positive? 
                # Wait. r - 1/d. If d increases, 1/d decreases, so r - 1/d increases.
                # So if new_r < 0, it means 1/d > r.
                # But d >= d_min = ceil(1/r) implies 1/d <= r.
                # So new_r should be >= 0.
                # Unless there's integer division error? No.
                # So this break is likely never hit or only if d_min was wrong.
                
            res = can_solve(new_r, d, K)
            if res is not None:
                return [d] + res
                
        return None

    # Find initial upper bound
    # We know {7, 56, 1736} works.
    # Let's verify and use 1736 as high.
    # But to be safe, let's run a quick greedy to find a valid K.
    # Or just set K_high = 2000.
    
    K_low = 1
    K_high = 2000 # We expect optimal to be < 2000
    
    # Check if K_high is valid
    # If not, increase it.
    while time.time() < deadline - 5.0:
        sol = can_solve(target, 0, K_high)
        if sol is not None:
            break
        K_high *= 2
        if K_high > 100000:
            # Fallback to greedy
            r = target
            denoms = []
            used = set()
            while r > 0:
                d = (r.denominator + r.numerator - 1) // r.numerator
                while d in used:
                    d += 1
                denoms.append(d)
                used.add(d)
                r -= Fraction(1, d)
            return denoms
            
    # Binary search for minimal K
    best_sol = None
    while K_low < K_high and time.time() < deadline - 2.0:
        K_mid = (K_low + K_high) // 2
        sol = can_solve(target, 0, K_mid)
        if sol is not None:
            best_sol = sol
            K_high = K_mid
        else:
            K_low = K_mid + 1
            
    if best_sol is None:
        # Try K_high again in case of time limit issues or logic error
        best_sol = can_solve(target, 0, K_high)
        
    if best_sol is None:
        # Final fallback: pure greedy
        r = target
        denoms = []
        used = set()
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            while d in used:
                d += 1
            denoms.append(d)
            used.add(d)
            r -= Fraction(1, d)
        return denoms
        
    return best_sol
