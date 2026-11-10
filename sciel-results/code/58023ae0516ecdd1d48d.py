"""Search-based program: find A,B with 3 distinct integers in [1,128] maximizing |A+B| with pairwise coprime sums."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin under 90s wall-clock
    
    # Precompute pairwise coprime compatibility for sums up to 256
    max_sum = 256
    coprime = [[False] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        for j in range(i, max_sum + 1):
            if gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True
    
    def is_pairwise_coprime_fast(vals):
        """Fast pairwise coprime check using precomputed table"""
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if not coprime[lst[i]][lst[j]]:
                    return False
        return True
    
    def score_of(A, B):
        s = {a + b for a in A for b in B}
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime_fast(s):
            return -1
        return len(s)
    
    # Start with the seed solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Strategy: Search for configurations with 9 distinct pairwise coprime sums
    # Key: all 9 sums a_i + b_j must be distinct AND pairwise coprime
    
    # Phase 1: Structured search with arithmetic progressions
    # A = [a, a+d1, a+2*d1], B = [b, b+d2, b+2*d2]
    # For 9 distinct sums: d1 != d2
    # For pairwise coprime: all 9 sums must be mutually coprime
    
    # Precompute valid (d1, d2) pairs that tend to produce coprime sums
    # Sums are: base, base+d2, base+2*d2, base+d1, base+d1+d2, base+d1+2*d2, base+2*d1, base+2*d1+d2, base+2*d1+2*d2
    # where base = a+b
    
    # Try many base values and difference pairs
    found_nine = False
    
    # First pass: systematic search over difference pairs
    # Focus on small differences first (more likely to stay in range)
    for d1 in range(1, 35):
        if time.time() > deadline:
            return [best_A, best_B]
        for d2 in range(1, 35):
            if time.time() > deadline:
                return [best_A, best_B]
            if d2 == d1:
                continue
            
            # Check if this difference pair can potentially yield 9 distinct sums
            # The 9 sums relative to base are: 0, d2, 2*d2, d1, d1+d2, d1+2*d2, 2*d1, 2*d1+d2, 2*d1+2*d2
            rel_sums = [0, d2, 2*d2, d1, d1+d2, d1+2*d2, 2*d1, 2*d1+d2, 2*d1+2*d2]
            
            # Check distinctness of relative sums
            if len(set(rel_sums)) < 9:
                continue
            
            # Now try different base values
            # base = a + b, where a in [1,128], b in [1,128]
            # a must satisfy: a >= 1, a+2*d1 <= 128, so a <= 128-2*d1
            # b must satisfy: b >= 1, b+2*d2 <= 128, so b <= 128-2*d2
            # base = a+b ranges from 2 to (128-2*d1)+(128-2*d2)
            
            max_a = 128 - 2 * d1
            max_b = 128 - 2 * d2
            if max_a < 1 or max_b < 1:
                continue
            
            # Try a sample of base values
            # For efficiency, try every 2nd or 3rd base value
            step = 2
            min_base = 2
            max_base = max_a + max_b
            
            for base in range(min_base, max_base + 1, step):
                if time.time() > deadline:
                    return [best_A, best_B]
                
                # Find a, b such that a+b = base, a <= max_a, b <= max_b, a>=1, b>=1
                a_min = max(1, base - max_b)
                a_max = min(max_a, base - 1)
                
                if a_min > a_max:
                    continue
                
                # Try a few values of a in the valid range
                a_try = a_min
                if a_try > a_max:
                    continue
                
                a = a_try
                b = base - a
                if b < 1 or b > max_b:
                    continue
                
                A = [a, a + d1, a + 2 * d1]
                B = [b, b + d2, b + 2 * d2]
                
                # Verify all elements in range
                if A[0] < 1 or A[2] > 128 or B[0] < 1 or B[2] > 128:
                    continue
                
                s = score_of(A, B)
                if s > best_score:
                    best_score = s
                    best_A = A[:]
                    best_B = B[:]
                    if s == 9:
                        found_nine = True
                        # Don't break immediately - keep searching for robustness
                        # but we can reduce search intensity
                        step = 3  # Skip more values
    
    # Phase 2: If we haven't found 9, do a more targeted random search
    if not found_nine and best_score < 9:
        import random
        random.seed(42)
        
        # Generate random sets with structure to increase chance of distinct sums
        for _ in range(200000):
            if time.time() > deadline:
                break
            
            # Generate A with some spread
            a1 = random.randint(1, 128)
            d1 = random.randint(2, 30)
            a2 = a1 + d1
            a3 = a1 + 2 * d1
            if a3 > 128:
                continue
            A = [a1, a2, a3]
            
            # Generate B with different spread
            b1 = random.randint(1, 128)
            d2 = random.randint(2, 30)
            if d2 == d1:
                d2 += 1
            b2 = b1 + d2
            b3 = b1 + 2 * d2
            if b3 > 128:
                continue
            B = [b1, b2, b3]
            
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A[:]
                best_B = B[:]
                if s == 9:
                    found_nine = True
                    break
    
    # Phase 3: If still not 9, try exhaustive search over small structured families
    if not found_nine and best_score < 9:
        # Try all combinations of small sets
        # Focus on sets with elements that are mostly odd or have specific prime factors
        candidates_a = []
        candidates_b = []
        
        # Generate candidate sets with good spread
        for a1 in range(1, 50):
            for d1 in range(2, 20):
                a2 = a1 + d1
                a3 = a1 + 2 * d1
                if a3 <= 128:
                    candidates_a.append([a1, a2, a3])
        
        for b1 in range(1, 50):
            for d2 in range(2, 20):
                b2 = b1 + d2
                b3 = b1 + 2 * d2
                if b3 <= 128:
                    candidates_b.append([b1, b2, b3])
        
        # Search over candidates
        for A in candidates_a:
            if time.time() > deadline:
                break
            for B in candidates_b:
                if time.time() > deadline:
                    break
                
                # Quick check: are the relative differences different?
                dA = A[1] - A[0]
                dB = B[1] - B[0]
                if dA == dB:
                    continue
                
                s = score_of(A, B)
                if s > best_score:
                    best_score = s
                    best_A = A[:]
                    best_B = B[:]
                    if s == 9:
                        found_nine = True
                        break
            if found_nine:
                break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
