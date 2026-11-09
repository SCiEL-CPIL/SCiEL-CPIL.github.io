import math
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill

    # Precompute primes up to 256
    is_prime = [True] * 257
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i * i, 257, i):
                is_prime[j] = False

    odds = [i for i in range(1, 129, 2)]
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5

    # Build adjacency: for each a in [1,128], list of b in [1,128] where a+b is prime
    valid_b_for_a = {}
    for a in range(1, 129):
        valid_b_for_a[a] = [b for b in range(1, 129) if is_prime[a + b]]
    
    # Precompute sets for fast intersection
    valid_b_set = {a: set(valid_b_for_a[a]) for a in range(1, 129)}

    # Strategy: Find A = {a1, a2, a3} (odd) and B = {b1, b2, b3} (even)
    # such that all 9 sums a_i + b_j are distinct primes.
    # If a_i are odd and b_j are even, a_i+b_j is odd. If it's prime, it's coprime to other odd primes.
    # So we just need 9 distinct primes.
    
    # We iterate over pairs (a1, a2) and find common valid b's.
    # To speed up, we can sort odds by "popularity" or just iterate.
    # Let's optimize by pre-computing common sets for pairs? 128*128/2 = 8000 pairs. 
    # Storing 8000 sets of size ~50 is fine memory-wise.
    
    # But actually, the triple loop with pruning is likely fast enough.
    # Let's refine the triple loop.
    
    for a1 in odds:
        if time.time() > deadline:
            break
        vb1_set = valid_b_set[a1]
        if len(vb1_set) < 3:
            continue
        for a2 in odds:
            if a2 <= a1:
                continue
            if time.time() > deadline:
                break
            vb2_set = valid_b_set[a2]
            common12 = vb1_set & vb2_set
            if len(common12) < 3:
                continue
            for a3 in odds:
                if a3 <= a2:
                    continue
                if time.time() > deadline:
                    break
                vb3_set = valid_b_set[a3]
                common123 = common12 & vb3_set
                if len(common123) < 3:
                    continue
                
                A = [a1, a2, a3]
                common123_list = sorted(common123)
                # Try to find 3 b's from common123_list with all 9 sums distinct
                # Since all sums are prime, they are automatically pairwise coprime.
                # We just need distinctness.
                
                # Heuristic: try smaller b's first? Or random?
                # Let's just iterate combinations.
                # Limit candidates to keep it fast if huge?
                # Max size of common123 is 128. Combinations(128,3) is ~350k.
                # This might be too slow if many triplets pass the filter.
                # But the filter `len(common123) < 3` is quite strict.
                
                # Optimization: If we find ANY triplet, we can return immediately 
                # if we know 9 is the max possible. 
                # Max distinct sums is 9. 
                # If we find 9 distinct primes, that's the global max.
                
                # Let's add a check: if len(common123) is large, pick a few random combos first?
                # Or just iterate.
                
                found = False
                # To speed up, we can check distinctness quickly.
                # Sums are a1+b, a2+b, a3+b for b in B.
                # For distinctness, we need a_i + b_j != a_k + b_l for (i,j) != (k,l).
                # This is equivalent to b_j - b_l != a_k - a_i.
                
                # Let's just use the set method for clarity, it's O(9) per combo.
                for B_cand in combinations(common123_list, 3):
                    B = list(B_cand)
                    sums = set()
                    distinct = True
                    for a in A:
                        for b in B:
                            s = a + b
                            if s in sums:
                                distinct = False
                                break
                            sums.add(s)
                        if not distinct:
                            break
                    if distinct and len(sums) == 9:
                        # Verify pairwise coprime (redundant for primes, but safe)
                        vals = list(sums)
                        ok = True
                        for i in range(9):
                            for j in range(i+1, 9):
                                if math.gcd(vals[i], vals[j]) != 1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            return A, B
    
    # If the prime search fails (unlikely, but possible if no 3x3 grid of distinct primes exists),
    # fallback to general search.
    # But 9 distinct primes in a 3x3 grid is very likely to exist.
    
    # Fallback search for any 9 distinct pairwise coprime sums
    # This is expensive, so we limit it.
    nums = list(range(1, 50))
    for A in combinations(nums, 3):
        if time.time() > deadline:
            break
        A = list(A)
        for B in combinations(nums, 3):
            if time.time() > deadline:
                break
            B = list(B)
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) > best_score:
                vals = list(sums)
                ok = True
                for i in range(len(vals)):
                    for j in range(i+1, len(vals)):
                        if math.gcd(vals[i], vals[j]) != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best_score = len(sums)
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return best_A, best_B
    
    return best_A, best_B
