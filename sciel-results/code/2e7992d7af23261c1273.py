"""
Search-based program: find A,B with 3 distinct integers in [1,128] maximizing |A+B| 
with pairwise coprime sums.

Strategy:
1. Precompute a pairwise coprime lookup table for sums up to 256.
2. Systematically search over arithmetic progression differences (d1, d2) for A and B.
3. For each pair of differences, check if the relative sums are distinct.
4. If distinct, iterate over valid base values (a+b) to find a configuration where 
   all 9 absolute sums are pairwise coprime.
5. Since 9 is the maximum possible cardinality, return the first valid solution found 
   or the best partial solution if none is found (though 9 is known to exist).
"""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin under 90s wall-clock
    
    # Precompute pairwise coprime compatibility for sums up to 256 (max 128+128)
    max_sum = 256
    # coprime[i][j] is True if gcd(i, j) == 1
    coprime = [[False] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(1, max_sum + 1):
            if gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True
    
    def is_pairwise_coprime_fast(vals):
        """Fast pairwise coprime check using precomputed table."""
        lst = list(vals)
        n = len(lst)
        for i in range(n):
            vi = lst[i]
            # Optimization: if vi is 1, it's coprime with everything
            if vi == 1:
                continue
            for j in range(i + 1, n):
                vj = lst[j]
                if vj == 1:
                    continue
                if not coprime[vi][vj]:
                    return False
        return True
    
    def score_of(A, B):
        s = {a + b for a in A for b in B}
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime_fast(s):
            return -1
        return len(s)
    
    # Start with a known good seed (from previous runs/inspiration)
    # The seed [1, 7, 13] and [4, 10, 16] yields sums:
    # 1+4=5, 1+10=11, 1+16=17
    # 7+4=11, 7+10=17, 7+16=23
    # 13+4=17, 13+10=23, 13+16=29
    # Distinct sums: {5, 11, 17, 23, 29}. Size 5.
    # Pairwise coprime?
    # 5,11,17,23,29 are all prime, so yes.
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Search for a configuration with 9 distinct pairwise coprime sums.
    # A = [a, a+d1, a+2*d1]
    # B = [b, b+d2, b+2*d2]
    # Sums are base + offset, where base = a+b.
    # Offsets: 0, d2, 2*d2, d1, d1+d2, d1+2*d2, 2*d1, 2*d1+d2, 2*d1+2*d2
    
    # We iterate over d1, d2. If d1 == d2, sums are not distinct (collisions).
    # We need the 9 offsets to be distinct.
    
    found_optimal = False
    
    # Limit d1, d2 to ensure a+2*d1 <= 128 and b+2*d2 <= 128 for some a,b >= 1.
    # Max d is roughly 63.
    max_d = 64
    
    for d1 in range(1, max_d):
        if time.time() > deadline:
            break
        for d2 in range(1, max_d):
            if time.time() > deadline:
                break
            if d1 == d2:
                continue
            
            offsets = [0, d2, 2*d2, d1, d1+d2, d1+2*d2, 2*d1, 2*d1+d2, 2*d1+2*d2]
            
            # Check if offsets are distinct
            if len(set(offsets)) != 9:
                continue
            
            # Now find a valid base = a+b such that all (base + offset) are in [2, 256]
            # and pairwise coprime.
            # a must be in [1, 128-2*d1]
            # b must be in [1, 128-2*d2]
            # min_base = 1+1 = 2
            # max_base = (128-2*d1) + (128-2*d2)
            
            min_base = 2
            max_base = (128 - 2 * d1) + (128 - 2 * d2)
            
            if max_base < min_base:
                continue
            
            # Try a subset of base values to save time, or all if fast enough.
            # There are at most ~250 base values. Checking 9 pairwise coprimes is cheap.
            # Total iterations: 64*64*250 ~ 1M checks. Very fast in Python with precomputed table.
            
            for base in range(min_base, max_base + 1):
                # Check if there exists a,b for this base
                # a >= 1, a <= 128-2*d1
                # b = base - a
                # b >= 1 => a <= base - 1
                # b <= 128-2*d2 => base - a <= 128-2*d2 => a >= base - (128-2*d2)
                
                a_min = max(1, base - (128 - 2 * d2))
                a_max = min(128 - 2 * d1, base - 1)
                
                if a_min > a_max:
                    continue
                
                # Pick a valid a (e.g., a_min)
                a = a_min
                b = base - a
                
                # Construct A and B
                A = [a, a + d1, a + 2 * d1]
                B = [b, b + d2, b + 2 * d2]
                
                # Verify bounds (should be satisfied by construction, but safe check)
                if A[0] < 1 or A[2] > 128 or B[0] < 1 or B[2] > 128:
                    continue
                
                # Calculate sums
                sums = [base + off for off in offsets]
                
                # Check pairwise coprime
                if is_pairwise_coprime_fast(sums):
                    # We found a valid set with 9 distinct sums
                    # Since 9 is the max possible, we can return immediately
                    return [A, B]
            
    # If the structured search didn't find 9 (which is unlikely), fall back to random search
    # or return the best found so far.
    import random
    random.seed(42)
    for _ in range(100000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        s = score_of(A, B)
        if s > best_score:
            best_score = s
            best_A = A
            best_B = B
            if s == 9:
                break
                
    return [best_A, best_B]
# EVOLVE-BLOCK-END
