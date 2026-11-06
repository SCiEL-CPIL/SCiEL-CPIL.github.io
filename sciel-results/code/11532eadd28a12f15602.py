"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

_DEADLINE = time.time() + 80  # 80s safety margin out of 90s

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _precompute_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    return [i for i in range(2, limit+1) if sieve[i]]

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            if _gcd(vals[i], vals[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    
    # Max sum is 128+128 = 256
    primes = set(_precompute_primes(256))
    
    # We want to find A, B with 3 distinct elements each in [1,128]
    # such that |A+B| is maximized and all distinct sums are pairwise coprime.
    
    # Best known: score 1.0 means 5 distinct sums.
    # Let's try to find 6, 7, 8, or 9 distinct pairwise-coprime sums.
    
    # Strategy: 
    # 1. Try to find 9 distinct sums all prime (best case)
    # 2. If not, try 8, 7, 6 distinct sums pairwise coprime
    
    # For efficiency, let's use a structured approach:
    # Fix A to be "spread out" to maximize distinct sums.
    
    # First, let's try a direct search with good candidates.
    # A good structure: A = {a, a+d1, a+d1+d2}, B = {b, b+e1, b+e1+e2}
    # with large gaps to avoid collisions.
    
    # Let's try: generate A candidates, then for each find best B.
    
    # To make this tractable, let's limit our search intelligently.
    
    best_score = 5  # current known
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    # Precompute all possible A sets? Too many. 
    # Instead, use heuristic: A should have elements spread out.
    
    # Let's try specific structured families:
    # Family 1: Arithmetic progressions
    # Family 2: Geometric-like spreads
    # Family 3: Random with pruning
    
    # Actually, let's do a smarter search:
    # For each candidate A (sampled), find B that maximizes distinct coprime sums.
    
    # Key insight: if all 9 sums are prime, we're done.
    # Let's first search for 9-prime-sum solutions.
    
    # Generate candidate A sets: focus on small values to keep sums in prime-friendly range
    # Primes up to 256: there are 54 primes.
    
    # Let's try: A = {a, a+p, a+q} where p, q are chosen to spread sums.
    
    # More efficient: iterate over A from a reasonable subset, and for each A,
    # try to find B greedily or via small search.
    
    # Let's limit A search to values in [1, 64] to keep computation manageable,
    # and use the time budget wisely.
    
    # Actually, with 128^3/6 ≈ 350k A sets, and for each we'd need to search B...
    # That's too slow. Let's be smarter.
    
    # Alternative: enumerate all pairs of "spread" sets.
    
    # Let me try a different approach: 
    # For maximum distinct sums, A and B should have no "additive structure" that causes collisions.
    # A collision occurs when a1+b1 = a2+b2, i.e., a1-a2 = b2-b1.
    # So we need all pairwise differences in A to be distinct from all pairwise differences in B.
    
    # For 3 elements, A has 3 pairwise differences (with signs), B has 3.
    # We need no overlap (except 0 which we don't have since elements are distinct).
    
    # Let's generate A with "unique difference set" and same for B.
    
    # For 3 elements {x, y, z} with x<y<z, differences are: y-x, z-y, z-x.
    # We need z-x != y-x and z-x != z-y (always true since x,y,z distinct and ordered)
    # and y-x != z-y (i.e., not an arithmetic progression).
    
    # So A and B should NOT be arithmetic progressions (to have 3 distinct differences).
    # But we need the difference sets of A and B to be disjoint.
    
    # Let's try:
    # A = {a1, a2, a3}, B = {b1, b2, b3}
    # Differences in A: dA1=a2-a1, dA2=a3-a2, dA3=a3-a1
    # Differences in B: dB1=b2-b1, dB2=b3-b2, dB3=b3-b1
    # Need: no element in {dA1,dA2,dA3} equals any in {dB1,dB2,dB3}
    # (and also no negative version, but since we consider absolute differences for collision)
    
    # Actually collision: a_i - a_j = b_k - b_l for i≠j, k≠l
    # So we need the signed difference sets to be disjoint.
    
    # This is getting complex. Let me just do a targeted search.
    
    # Approach: 
    # 1. Generate candidate A sets (spread out, not AP)
    # 2. For each A, generate candidate B sets
    # 3. Check if all 9 sums are distinct and pairwise coprime
    # 4. Track best
    
    # To make this feasible, let's limit the search space.
    
    # Let's try A values in [1, 80] and B values in [1, 80]
    # with the constraint that sums are in [2, 160] where there are many primes.
    
    # Number of A sets from [1,80]: C(80,3) = 82160
    # For each, trying all B sets is too much.
    
    # Better: for each A, greedily construct B.
    
    # Even better: let's just try a moderate number of random/structured candidates.
    
    # Let me try a specific promising structure:
    # A = {2, 5, 11}, B = {3, 7, 13}
    # Sums: 5,9,15, 8,12,18, 14,18,24 -> not all prime, has duplicates
    
    # Let me try to find 9 distinct prime sums.
    
    # For 9 distinct sums, we need:
    # All a_i + b_j distinct for i,j in {1,2,3}
    # All 9 sums prime
    
    # Let's search more systematically but with time limit.
    
    # Generate A from a subset: use values that are "well-spread"
    # Let's try A where elements are roughly in [1,40] and B in [40,80] or similar.
    
    # Actually, let me just do a randomized search with good initialization.
    
    import random
    random.seed(42)
    
    # Precompute: for quick checking
    # We'll check: 9 distinct sums, all prime
    
    found_9 = False
    found_8 = False
    best_8 = None
    best_7 = None
    best_6 = None
    
    # Let's try structured search first
    # A: pick 3 from [1, 60], B: pick 3 from [1, 60]
    # but with time limit
    
    # For speed, let's precompute a set of valid (A,B) pairs where all sums are prime.
    # This is still huge, so let's sample.
    
    # Actually, let me think about this differently.
    # The number of primes up to 256 is 54.
    # We need 9 distinct primes that can be written as a_i + b_j.
    
    # Let me try: pick 9 target primes, then see if they can be partitioned into a 3x3 sum grid.
    
    # A 3x3 sum grid: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    # The structure: (a_i + b_j) - (a_i + b_k) = b_j - b_k
    # And (a_i + b_j) - (a_k + b_j) = a_i - a_k
    
    # So the 9 sums form a "grid" in the additive sense.
    
    # Let me just do a direct search with time limit.
    
    # Strategy: 
    # - Try all A from a smaller range, say [1, 50]
    # - For each A, try a limited set of B candidates
    # - Check pairwise coprime condition
    
    # C(50,3) = 19600 A sets. For each, trying say 1000 B candidates = 19.6M checks.
    # Each check is 9 GCD computations. Should be feasible in 80s.
    
    # Actually, let's be even more targeted.
    
    # Let me try: A in [1,40], B in [1,40], but only check if all sums are prime first (fast),
    # then check pairwise coprime (which is automatic if all prime).
    
    # If all 9 sums are prime and distinct, we have score 9/5 = 1.8
    
    # Let's search for this.
    
    # To speed up: precompute which triples of numbers have all pair-sums prime with another triple.
    
    # Actually, let me just do it directly with time check.
    
    def check_AB(A, B):
        """Returns (num_distinct, all_pairwise_coprime)"""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        vals = list(sums)
        if len(vals) == 0:
            return 0, False
        # Check pairwise coprime
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if _gcd(vals[i], vals[j]) != 1:
                    return len(vals), False
        return len(vals), True
    
    # Search for 9 distinct prime sums
    # A from [1, 60], B from [1, 60]
    # But C(60,3) = 34220, and for each we'd need to check many B's.
    
    # Better approach: for each A, find B such that all 9 sums are prime.
    
    # Let's limit: A from [1, 50], and for each A, try B from a precomputed list of "good" B's.
    
    # Actually, let me just iterate A over [1,45] and B over [1,45] with early termination.
    
    # C(45,3) = 14190. For each A, I'll try a subset of B's.
    
    # Let me precompute all B triples from [1,45] and for each A, check against all B's.
    # That's 14190 * 14190 ≈ 200M, too slow.
    
    # Alternative: for each A, generate B greedily.
    
    # Or: fix B and search A, or use a meet-in-the-middle.
    
    # Let me try a different approach: 
    # Enumerate all B triples from [1, 50] (19600 of them).
    # For each B, find A triples such that all sums are prime.
    
    # Still too slow for full enumeration.
    
    # Let me just do a randomized search with many iterations.
    
    # Actually, let me think about what values work.
    
    # Known good: A={1,7,13}, B={4,10,16} gives sums {5,11,17,23,29} (5 primes)
    
    # Let me try to extend this pattern.
    
    # Try: A={1,7,13}, vary B
    # Sums with B={b1,b2,b3}: {1+b1,1+b2,1+b3, 7+b1,7+b2,7+b3, 13+b1,13+b2,13+b3}
    
    # For all 9 to be prime and distinct:
    # Need b1,b2,b3 such that 1+b_i, 7+b_i, 13+b_i are all prime for i=1,2,3
    # And all 9 values distinct.
    
    # 1+b_i prime, 7+b_i prime, 13+b_i prime.
    # b_i+1, b_i+7, b_i+13 all prime.
    
    # Let's find b values where b+1, b+7, b+13 are all prime.
    
    # b+1 prime, b+7 prime, b+13 prime.
    # b+7 - (b+1) = 6, b+13 - (b+7) = 6, b+13 - (b+1) = 12.
    # So we need 3 primes in arithmetic progression with common difference 6.
    
    # Primes in AP with diff 6: (p, p+6, p+12) all prime.
    # p=5: 5,11,17 all prime! b=4: 4+1=5, 4+7=11, 4+13=17. Yes!
    # p=7: 7,13,19 all prime! b=6: 6+1=7, 6+7=13, 6+13=19. Yes!
    # p=11: 11,17,23 all prime! b=10: 10+1=11, 10+7=17, 10+13=23. Yes!
    # p=13: 13,19,25 - 25 not prime.
    # p=17: 17,23,29 all prime! b=16: 16+1=17, 16+7=23, 16+13=29. Yes!
    # p=19: 19,25,31 - no.
    # p=23: 23,29,35 - no.
    # p=31: 31,37,43 all prime! b=30: 30+1=31, 30+7=37, 30+13=43. Yes!
    # p=37: 37,43,49 - no.
    # p=43: 43,49,55 - no.
    # p=47: 47,53,59 all prime! b=46: 46+1=47, 46+7=53, 46+13=59. Yes!
    # p=53: 53,59,65 - no.
    # p=61: 61,67,73 all prime! b=60: 60+1=61, 60+7=67, 60+13=73. Yes!
    # p=67: 67,73,79 all prime! b=66: 66+1=67, 66+7=73, 66+13=79. Yes!
    # p=73: 73,79,85 - no.
    # p=79: 79,85,91 - no.
    # p=83: 83,89,95 - no.
    # p=89: 89,95,101 - no.
    # p=97: 97,103,109 all prime! b=96: 96+1=97, 96+7=103, 96+13=109. Yes!
    # p=101: 101,107,113 all prime! b=100: 100+1=101, 100+7=107, 100+13=113. Yes!
    
    # So valid b values for A={1,7,13}: b in {4, 6, 10, 16, 30, 46, 60, 66, 96, 100, ...}
    
    # Now I need 3 distinct b values from this set, all in [1,128].
    # b in {4, 6, 10, 16, 30, 46, 60, 66, 96, 100}
    
    # Let's pick b1=4, b2=6, b3=10.
    # Sums: 
    # 1+4=5, 1+6=7, 1+10=11
    # 7+4=11, 7+6=13, 7+10=17
    # 13+4=17, 13+6=19, 13+10=23
    # Distinct sums: {5,7,11,13,17,19,23} - that's 7 distinct values!
    # Check pairwise coprime: all prime, so yes!
    
    # Wait, 11 appears twice (1+10 and 7+4), 17 appears twice (7+10 and 13+4).
    # Distinct: {5,7,11,13,17,19,23} = 7 values. All prime, so pairwise coprime.
    
    # That gives score 7/5 = 1.4!
    
    # Can we do better? Try b1=4, b2=10, b3=16:
    # 1+4=5, 1+10=11, 1+16=17
    # 7+4=11, 7+10=17, 7+16=23
    # 13+4=17, 13+10=23, 13+16=29
    # Distinct: {5,11,17,23,29} = 5 values. Worse.
    
    # Try b1=4, b2=6, b3=16:
    # 1+4=5, 1+6=7, 1+16=17
    # 7+4=11, 7+6=13, 7+16=23
    # 13+4=17, 13+6=19, 13+16=29
    # Distinct: {5,7,11,13,17,19,23,29} = 8 values! All prime!
    
    # Check: 5,7,11,13,17,19,23,29 - all prime, so pairwise coprime.
    # Score: 8/5 = 1.6!
    
    # Can we get 9? Try b1=4, b2=6, b3=30:
    # 1+4=5, 1+6=7, 1+30=31
    # 7+4=11, 7+6=13, 7+30=37
    # 13+4=17, 13+6=19, 13+30=43
    # Distinct: {5,7,11,13,17,19,31,37,43} = 9 values! All prime!
    
    # Check pairwise coprime: all prime, so yes!
    # A={1,7,13}, B={4,6,30}
    # All in [1,128]: yes.
    # All distinct in A: 1,7,13 distinct. In B: 4,6,30 distinct.
    
    # Score: 9/5 = 1.8!
    
    # Let me verify:
    # A+B = {1+4, 1+6, 1+30, 7+4, 7+6, 7+30, 13+4, 13+6, 13+30}
    #     = {5, 7, 31, 11, 13, 37, 17, 19, 43}
    # All distinct: 5,7,11,13,17,19,31,37,43 - yes, 9 distinct values.
    # All prime: yes.
    # Pairwise coprime: yes (all prime).
    
    # This is great! Let me return this.
    
    return [[1, 7, 13], [4, 6, 30]]
# EVOLVE-BLOCK-END
