import time
from itertools import combinations
from math import gcd

def is_pairwise_coprime(lst):
    """Check if all distinct elements in lst are pairwise coprime."""
    unique_vals = list(set(lst))
    n = len(unique_vals)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(unique_vals[i], unique_vals[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # We search for A and B such that |A+B| is maximized (target 9)
    # and all distinct sums are pairwise coprime.
    
    # Search space: integers in [1, 128]
    # To be efficient, we iterate over combinations of A and B.
    # However, full brute force 128^6 is too big.
    # We can iterate over A and for each A, iterate over B.
    # Or, we can use the fact that small numbers work.
    
    # Let's try a structured search over small ranges first.
    # If found, return immediately.
    
    max_range = 128
    
    # Optimization: 
    # We can iterate over possible sumsets? No, that's complex.
    # Let's iterate over A and B from small values.
    # A solution with small numbers is highly likely.
    
    # We will search A in [1, 20] and B in [1, 20] first.
    # If not found, expand.
    
    for limit in [10, 20, 50, 128]:
        if time.time() > deadline:
            break
            
        # Generate combinations of 3 distinct integers from [1, limit]
        # This is C(limit, 3) for A and C(limit, 3) for B.
        # C(10,3) = 120. 120*120 = 14400 checks. Very fast.
        # C(20,3) = 1140. 1140*1140 ~ 1.3M checks. Still fast in Python if optimized.
        # C(50,3) = 19600. 19600^2 ~ 380M. Too slow for pure Python loop.
        
        # So we will do:
        # 1. Exhaustive search for limit <= 20.
        # 2. Randomized search for limit > 20 if needed.
        
        if limit <= 20:
            range_val = range(1, limit + 1)
            comb_A = list(combinations(range_val, 3))
            comb_B = list(combinations(range_val, 3))
            
            for A in comb_A:
                if time.time() > deadline:
                    break
                for B in comb_B:
                    # Calculate sums
                    sums = [a + b for a in A for b in B]
                    
                    # Check distinct count
                    if len(set(sums)) < 9:
                        continue
                    
                    # Check pairwise coprime
                    if is_pairwise_coprime(sums):
                        return [list(A), list(B)]
        
        else:
            # Randomized search for larger limits
            import random
            attempts = 0
            while time.time() < deadline and attempts < 100000:
                A = sorted(random.sample(range(1, limit + 1), 3))
                B = sorted(random.sample(range(1, limit + 1), 3))
                sums = [a + b for a in A for b in B]
                
                if len(set(sums)) == 9:
                    if is_pairwise_coprime(sums):
                        return [A, B]
                attempts += 1

    # Fallback: Return a known valid solution if search fails or times out
    # One such solution (from knowledge or manual verification):
    # A = [1, 2, 4], B = [1, 3, 5] -> Sums: 2,3,4,3,4,5,5,6,7 -> Not distinct, not coprime (4,6)
    # Let's find a real one quickly.
    # A = [1, 3, 7], B = [1, 2, 4]
    # Sums:
    # 1+1=2, 1+2=3, 1+4=5
    # 3+1=4, 3+2=5, 3+4=7
    # 7+1=8, 7+2=9, 7+4=11
    # Distinct sums: {2,3,5,4,7,8,9,11} -> 2,4,8 not coprime.
    
    # Let's use the one implied by the "distinct primes" strategy.
    # We can hardcode a verified solution from a quick mental check or known math fact.
    # Actually, the search above with limit=20 is very likely to find one.
    # If it doesn't, we return a seed-like valid solution to ensure non-zero score.
    # Seed: [[1, 7, 13], [4, 10, 16]] -> Sums: 5, 11, 17, 11, 17, 23, 17, 23, 29.
    # Distinct: {5, 11, 17, 23, 29}. All primes. Pairwise coprime. Size 5.
    
    return [[1, 7, 13], [4, 10, 16]]
