import time
import math

def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin
    
    # Target XOR: Squarefree kernel of 210
    # 210 = 2 * 3 * 5 * 7
    # Map primes to bits: 2->0, 3->1, 5->2, 7->3
    # Target mask = 1<<0 | 1<<1 | 1<<2 | 1<<3 = 15
    
    target_mask = 15
    
    # Generate candidates
    # We restrict to numbers with prime factors in {2, 3, 5, 7} to keep the state space small
    # and ensure no "extra" primes are introduced that need cancellation.
    # If a solution exists in this set, it is valid.
    # Given the density of smooth numbers, a small solution is likely.
    
    candidates = []
    for n in range(211, 10211):
        t = n
        mask = 0
        
        if t % 2 == 0:
            mask |= 1
            t //= 2
            while t % 2 == 0:
                t //= 2
        if t % 3 == 0:
            mask |= 2
            t //= 3
            while t % 3 == 0:
                t //= 3
        if t % 5 == 0:
            mask |= 4
            t //= 5
            while t % 5 == 0:
                t //= 5
        if t % 7 == 0:
            mask |= 8
            t //= 7
            while t % 7 == 0:
                t //= 7
        
        if t == 1:
            candidates.append((n, mask))
            
        if time.time() > deadline - 10.0:
            break
    
    # Sort by value
    candidates.sort()
    
    # DP: dp[xor_val] = list of numbers
    dp = {0: []}
    
    for n, m in candidates:
        if time.time() > deadline:
            break
            
        updates = {}
        
        for x, nums in dp.items():
            nx = x ^ m
            if nx not in dp:
                updates[nx] = nums + [n]
        
        for nx, new_nums in updates.items():
            dp[nx] = new_nums
            
        if 15 in dp:
            result = dp[15]
            # Verify
            prod = 1
            for r in result:
                prod *= r
            total = prod * 210
            sqrt_t = math.isqrt(total)
            if sqrt_t * sqrt_t == total:
                return result
            else:
                # Fallback if verification fails (should not happen)
                return [840]
                
    # If no solution found in restricted set, fallback to [840] which is known valid
    return [840]
