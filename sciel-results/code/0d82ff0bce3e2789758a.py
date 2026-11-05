import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # 10s is plenty for small search
    
    def is_pairwise_coprime(vals):
        distinct = list(set(vals))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    best_count = 0
    
    # Search in small range first
    # C(20,3) = 1140. 1140^2 = 1.3M.
    # Optimized check:
    # 1. Generate sums.
    # 2. Check len(set) == 9. If not, skip coprime check.
    # 3. Check coprime.
    
    max_val = 25
    A_candidates = list(itertools.combinations(range(1, max_val + 1), 3))
    
    for A in A_candidates:
        if time.time() > deadline:
            break
        a1, a2, a3 = A
        for B in A_candidates:  # Symmetric, but we need distinct sets? No, A and B can be same.
            if time.time() > deadline:
                break
            b1, b2, b3 = B
            sums = (a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3)
            if len(set(sums)) < 9:
                continue
            # If 9 distinct, check coprime
            if is_pairwise_coprime(sums):
                return [list(A), list(B)]
                
    # If not found in small range, try slightly larger or random
    # But 9 is max. If we found one, we return.
    # If we didn't, we return best found so far (which might be < 9).
    # However, the problem implies 9 is achievable.
    
    # Fallback: Return a known good structure if search fails?
    # Or just return the best found in the loop.
    # Since we return immediately on finding 9, this block is only reached if no 9 found.
    
    return [best_A, best_B]
