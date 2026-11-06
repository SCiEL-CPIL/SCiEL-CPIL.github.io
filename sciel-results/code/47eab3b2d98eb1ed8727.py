"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    t0 = time.time()
    TL = 55.0
    
    target = Fraction(5, 31)
    
    # Known baseline: 1/8 + 1/28 + 1/1736 = 5/31
    best_solution = [8, 28, 1736]
    best_max_den = 1736
    
    # Try to find better solutions using direct algebraic enumeration for small term counts
    # For 3 terms: 1/a + 1/b + 1/c = 5/31
    # We can iterate over a, b and solve for c exactly.
    
    # Strategy:
    # 1. Iterate over possible max denominator K from small to 1736 (or binary search).
    # 2. For each K, check if there's a solution with all denominators <= K.
    #    Since we want to minimize K, we can iterate K from 1 upwards? No, that's slow.
    #    Better: Binary search on K? But the property "exists solution with max <= K" is monotonic.
    #    If a solution exists with max <= K, it also exists with max <= K+1.
    #    So we can binary search for the minimal K.
    
    # However, checking "exists solution with max <= K" is the hard part.
    # For 3 terms:
    # 1/a + 1/b + 1/c = 5/31
    # 1/c = 5/31 - 1/a - 1/b = (5ab - 31b - 31a) / (31ab)
    # c = 31ab / (5ab - 31b - 31a)
    # We need c to be an integer, c > b > a, and c <= K.
    # Also a, b, c must be distinct (implied by strict inequality if we enforce a < b < c).
    
    # Let's write a function to check if a 3-term solution exists with max <= K.
    def check_3_terms(K):
        # a must be at least ceil(31/5) = 7? 
        # 1/a <= 5/31 => a >= 31/5 = 6.2 => a >= 7.
        # Also 3/a > 5/31 => a < 93/5 = 18.6 => a <= 18.
        # So a is in [7, 18].
        
        # For a given a, 1/b + 1/c = 5/31 - 1/a = (5a - 31) / (31a)
        # Let R = (5a - 31) / (31a). Note R > 0 requires 5a > 31 => a >= 7.
        # 1/b < R => b > 1/R = 31a / (5a - 31).
        # Also 2/b > R (since b < c, so 1/b > 1/c, so 1/b + 1/c < 2/b) => b < 2/R = 62a / (5a - 31).
        # So b is in (31a/(5a-31), 62a/(5a-31)].
        # Also b <= K.
        
        for a in range(7, 19):
            num = 5 * a - 31
            if num <= 0:
                continue
            den = 31 * a
            # R = num / den
            # b_min = floor(den/num) + 1
            b_min = den // num + 1
            # b_max = floor(2*den/num)
            b_max = (2 * den) // num
            
            # Also b <= K
            b_max = min(b_max, K)
            
            if b_min > b_max:
                continue
                
            for b in range(b_min, b_max + 1):
                # c = 1 / (R - 1/b) = 1 / ((num*b - den) / (den*b)) = den*b / (num*b - den)
                num_c = den * b
                den_c = num * b - den
                if den_c <= 0:
                    continue
                if num_c % den_c != 0:
                    continue
                c = num_c // den_c
                if c <= b:
                    continue
                if c > K:
                    continue
                # Check distinctness: a < b < c ensures distinct.
                # Check sum: 1/a + 1/b + 1/c == 5/31
                # This is guaranteed by construction if c is integer.
                # But let's verify to be safe.
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    return [a, b, c]
        return None

    # What about 4 terms?
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # This is harder to enumerate directly.
    # But maybe a 4-term solution has a smaller max denominator than the best 3-term solution?
    # Let's first find the optimal 3-term solution.
    
    # Binary search for optimal 3-term K
    low = 7
    high = best_max_den
    
    # We know 1736 works. Is there a smaller one?
    # Let's just iterate K from 7 upwards until we find one? 
    # The range is small (7 to 1736), so linear scan might be okay if check_3_terms is fast.
    # check_3_terms is O(12 * K) in worst case, but actually the range for b is small.
    # For a=7, b in (31*7/(35-31), 62*7/(35-31)] = (217/4, 434/4] = (54.25, 108.5] -> 55..108. ~50 values.
    # Total checks per K is small.
    
    # Actually, we can just loop K from 7 to 1736 and return the first one found.
    # Since we want to minimize K, the first one found in increasing K is the optimum for 3 terms.
    
    found_3 = False
    for K in range(7, best_max_den + 1):
        if time.time() - t0 > TL:
            break
        sol = check_3_terms(K)
        if sol:
            if max(sol) < best_max_den:
                best_max_den = max(sol)
                best_solution = sol
            found_3 = True
            break # Found minimal K for 3 terms
    
    # Now, can we do better with 4 terms?
    # If the optimal 3-term K is, say, 100, can we find a 4-term solution with K < 100?
    # Let's try to find any solution with K < best_max_den using a DFS with pruning.
    # But DFS is slow.
    # Maybe we can extend the algebraic approach?
    # 1/d = 5/31 - 1/a - 1/b - 1/c
    # This requires iterating a, b, c.
    # For a fixed K, a is in [7, ...], b is bounded, c is bounded.
    # The number of (a,b,c) triples might be manageable for small K.
    
    # Let's try to search for 4-term solutions with K < best_max_den.
    # We'll iterate K from 7 to best_max_den - 1.
    # For each K, check if a 4-term solution exists.
    
    def check_4_terms(K):
        # a < b < c < d <= K
        # 1/a + 1/b + 1/c + 1/d = 5/31
        # a >= 7
        # 4/a > 5/31 => a < 124/5 = 24.8 => a <= 24
        
        for a in range(7, 25):
            if time.time() - t0 > TL:
                return None
            R1 = target - Fraction(1, a)
            # 1/b + 1/c + 1/d = R1
            # b > 1/R1, 3/b > R1 => b < 3/R1
            b_min = int(1.0 / float(R1)) + 1
            b_max = min(int(3.0 / float(R1)), K)
            
            if b_min > b_max:
                continue
                
            for b in range(b_min, b_max + 1):
                if time.time() - t0 > TL:
                    return None
                R2 = R1 - Fraction(1, b)
                # 1/c + 1/d = R2
                # c > 1/R2, 2/c > R2 => c < 2/R2
                c_min = int(1.0 / float(R2)) + 1
                c_max = min(int(2.0 / float(R2)), K)
                
                if c_min > c_max:
                    continue
                
                for c in range(c_min, c_max + 1):
                    if time.time() - t0 > TL:
                        return None
                    R3 = R2 - Fraction(1, c)
                    if R3 <= 0:
                        continue
                    # d = 1 / R3
                    # d must be integer, d > c, d <= K
                    num_d, den_d = R3.numerator, R3.denominator
                    if num_d == 1:
                        d = den_d
                        if d > c and d <= K:
                            # Verify
                            if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) + Fraction(1, d) == target:
                                return [a, b, c, d]
                    # If num_d != 1, d is not integer.
        return None

    # Search for 4-term solution with K < best_max_den
    # To save time, only do this if best_max_den is small enough that the search space is manageable.
    # Or just try it for a few K values.
    
    if best_max_den < 5000: # Safety limit for search space
        for K in range(7, best_max_den):
            if time.time() - t0 > TL:
                break
            sol = check_4_terms(K)
            if sol:
                if max(sol) < best_max_den:
                    best_max_den = max(sol)
                    best_solution = sol
                break # Found minimal K for 4 terms (assuming we scan K upwards)
            # Note: check_4_terms is expensive. We might not be able to scan all K.
            # Let's add a time check inside the loop or break early.
            # Actually, the loop above breaks if time > TL.
            
    return best_solution
# EVOLVE-BLOCK-END
