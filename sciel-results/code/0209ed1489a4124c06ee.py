import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 512
    
    # Strategy: Find the number L in [1, N] that maximizes the number of divisors.
    # Return all divisors of L.
    # Why? If A is the set of divisors of L, then for any a, b in A, 
    # a | L and b | L implies lcm(a, b) | L.
    # Since L <= N, lcm(a, b) <= L <= N.
    # This satisfies the constraint.
    # This is a structural change from maximum clique search to number theoretic construction.
    
    max_div_count = 0
    best_L = 1
    divisors_of_best_L = [1]
    
    for L in range(1, N + 1):
        # Count divisors of L
        count = 0
        for d in range(1, int(L**0.5) + 1):
            if L % d == 0:
                count += 1
                if d != L // d:
                    count += 1
        
        if count > max_div_count:
            max_div_count = count
            best_L = L
            # Compute divisors
            divisors_of_best_L = []
            for d in range(1, int(L**0.5) + 1):
                if L % d == 0:
                    divisors_of_best_L.append(d)
                    if d != L // d:
                        divisors_of_best_L.append(L // d)
            
            # Early exit if we hit a known high bound? 
            # For N=512, max divisors is likely around 20-30.
            # 240 has 20 divisors. 360 has 24. 480 has 28? 
            # 480 = 2^5 * 3 * 5 -> (5+1)(1+1)(1+1) = 24.
            # 360 = 2^3 * 3^2 * 5 -> (3+1)(2+1)(1+1) = 24.
            # 240 = 2^4 * 3 * 5 -> (4+1)(1+1)(1+1) = 20.
            # 336 = 2^4 * 3 * 7 -> 20.
            # 420 = 2^2 * 3 * 5 * 7 -> (2+1)(1+1)(1+1)(1+1) = 24.
            # 480 = 2^5 * 3 * 5 -> 24.
            # 360 is good.
            # Is there a number with more?
            # 2^a * 3^b * 5^c ...
            # 2^5 * 3^1 * 5^1 = 480 (24 divs)
            # 2^4 * 3^2 * 5^1 = 720 > 512.
            # 2^3 * 3^2 * 5^1 = 360 (24 divs)
            # 2^2 * 3^2 * 5^1 = 180 (18 divs)
            # 2^5 * 3^2 = 288 (18 divs)
            # 2^4 * 3^1 * 5^1 = 240 (20 divs)
            # 2^3 * 3^1 * 5^1 * 7^1 = 840 > 512.
            # 2^2 * 3^1 * 5^1 * 7^1 = 420 (24 divs)
            # 2^1 * 3^1 * 5^1 * 7^1 * 11^1 = 2310 > 512.
            
            # So 24 seems to be the max.
            # Let's just find the one with max divisors.
            
            # We can break early if we find a number with divisors count >= 24? 
            # No, we want to be sure. But 512 is small.
            # We can just iterate all.
            
            # Optimization: if max_div_count is already high, we might not need to check all.
            # But 512 iterations is trivial.
            
    return divisors_of_best_L
