"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations, product

def _check_pairwise_coprime(values):
    """Check if all distinct values are pairwise coprime."""
    vals = list(set(values))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return (cardinality, is_valid) for the sumset A+B."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    if not _check_pairwise_coprime(distinct):
        return (0, False)
    return (len(distinct), True)

def _fast_evaluate(A, B, coprime_cache):
    """Faster evaluation using precomputed coprime info."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct = list(sums)
    # Check pairwise coprime
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            x, y = distinct[i], distinct[j]
            if x > y:
                x, y = y, x
            if (x, y) not in coprime_cache:
                coprime_cache[(x, y)] = (gcd(x, y) == 1)
            if not coprime_cache[(x, y)]:
                return (0, False)
    return (len(distinct), True)

def _precompute_coprime_pairs(max_val=256):
    """Precompute which pairs up to max_val are coprime."""
    coprime = {}
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            coprime[(i, j)] = (gcd(i, j) == 1)
    return coprime

def _find_valid_sumsets(coprime, min_sum=2, max_sum=256):
    """
    Find all sets of 9 distinct integers in [min_sum, max_sum] that are pairwise coprime.
    These represent potential A+B sumsets with all 9 sums distinct.
    """
    valid_sumsets = []
    
    # First, find all numbers that can be part of a pairwise coprime set
    # A number x can be in a pairwise coprime set if it's coprime to at least 8 other numbers
    # in the range that are also mutually coprime
    
    # Strategy: use backtracking to build pairwise coprime sets of size 9
    # This is expensive, so we'll be smart about it
    
    # Actually, let's think differently. We need to find A={a1,a2,a3} and B={b1,b2,b3}
    # such that all 9 sums a_i+b_j are distinct and pairwise coprime.
    
    # Key insight: if a_i + b_j = a_k + b_l for (i,j) != (k,l), then a_i - a_k = b_l - b_j
    # So we need the difference structure to avoid collisions.
    
    # For 9 distinct sums, we need:
    # - All differences a_i - a_j (i!=j) must be different from all differences b_k - b_l (k!=l) in absolute value
    # - More precisely: |a_i - a_j| != |b_k - b_l| for all i!=j, k!=l
    #   Wait, that's not quite right. We need a_i - a_k != b_l - b_j for all i!=k, j!=l
    #   This means the set {a_i - a_k : i!=k} and {b_l - b_j : j!=l} must be disjoint
    
    # Let's enumerate possible difference structures
    # For A = {a1, a2, a3} with a1 < a2 < a3:
    #   d1 = a2 - a1, d2 = a3 - a2, d3 = a3 - a1 = d1 + d2
    # For B = {b1, b2, b3} with b1 < b2 < b3:
    #   e1 = b2 - b1, e2 = b3 - b2, e3 = b3 - b1 = e1 + e2
    
    # The 9 sums are:
    #   a1+b1, a1+b2, a1+b3
    #   a2+b1, a2+b2, a2+b3
    #   a3+b1, a3+b2, a3+b3
    
    # For all to be distinct, we need no two to be equal.
    # a_i + b_j = a_k + b_l iff a_i - a_k = b_l - b_j
    # The possible differences in A (with sign): ±d1, ±d2, ±d3
    # The possible differences in B (with sign): ±e1, ±e2, ±e3
    # We need: {±d1, ±d2, ±d3} ∩ {±e1, ±e2, ±e3} = ∅
    # i.e., d1, d2, d3, e1, e2, e3 are all distinct in absolute value
    
    return valid_sumsets  # We'll build this differently

def _backtrack_find_solution(coprime, max_val=128):
    """
    Use backtracking on the difference structure to find A and B.
    
    We search over possible difference pairs (d1, d2) for A and (e1, e2) for B
    such that all 6 absolute differences are distinct, then check if the resulting
    sums are pairwise coprime.
    """
    solutions = []
    
    # Possible differences for 3-element sets in [1, 128]
    # If a1 >= 1, a3 <= 128, then d1 + d2 = a3 - a1 <= 127
    # d1 >= 1, d2 >= 1, so d1 + d2 <= 127
    
    # We'll iterate over possible (d1, d2) for A and (e1, e2) for B
    # and check if the resulting 9 sums are pairwise coprime
    
    # To make this efficient, we'll precompute which triples of differences work
    valid_diff_pairs = []
    for d1 in range(1, 128):
        for d2 in range(1, 128 - d1 + 1):
            d3 = d1 + d2
            valid_diff_pairs.append((d1, d2, d3))
    
    # Now for each pair of difference structures, check if they're compatible
    # and if we can find actual values that make the sums pairwise coprime
    
    # This is still O(n^2) where n ~ 127*127/2 ~ 8000, so ~64 million pairs
    # That's too many. Let's be smarter.
    
    # Alternative: fix A first, then search for B
    # For a given A = {a1, a2, a3}, the sums are:
    #   a1+b, a2+b, a3+b for each b in B
    # We need all 9 to be pairwise coprime.
    
    # Let's try a more direct approach: enumerate A and B with constraints
    
    # Key constraint: for the 9 sums to be pairwise coprime, no sum can share a factor
    # with any other sum.
    
    # Let's try small values first and use the coprime check
    best = None
    best_score = 0
    
    # Precompute coprime relationships for efficiency
    # coprime[x][y] = True if gcd(x,y) == 1
    
    # Try structured approach: use prime-related numbers
    # If all sums are primes, they're automatically pairwise coprime (except for 2)
    # But getting 9 primes as sums of two sets of 3 is hard
    
    # Let's try: A = {1, p, q} where p, q are chosen so that
    # 1+b1, 1+b2, 1+b3, p+b1, p+b2, p+b3, q+b1, q+b2, q+b3 are all pairwise coprime
    
    # Simplest case: try to make all sums odd (avoid factor 2)
    # and then check other factors
    
    # Let's do a targeted search
    rng = random.Random(12345)
    
    # Phase 1: Try to find A, B with 9 distinct pairwise coprime sums
    # Use the difference constraint to prune
    
    # For each valid (d1, d2) for A:
    #   For each valid (e1, e2) for B with distinct absolute differences:
    #     Try to find a1, b1 such that all sums are in [2, 256] and pairwise coprime
    
    # This is still expensive. Let's try a different angle.
    
    # Direct search with pruning:
    # Pick a1, a2, a3 (sorted), then pick b1, b2, b3 (sorted)
    # Check if all 9 sums are distinct and pairwise coprime
    
    # To speed up, precompute for each value v, which other values are coprime to it
    coprime_to = [set() for _ in range(257)]
    for i in range(2, 257):
        for j in range(i + 1, 257):
            if gcd(i, j) == 1:
                coprime_to[i].add(j)
                coprime_to[j].add(i)
    
    # Now search
    # We'll iterate over A in a structured way
    # Start with small values to keep sums small (more likely to be coprime)
    
    # Try A = {1, x, y} with 1 < x < y <= 128
    # Then B = {1, u, v} with 1 < u < v <= 128
    # Sums: 2, 1+u, 1+v, x+1, x+u, x+v, y+1, y+u, y+v
    
    # For these to be pairwise coprime:
    # 2 must be coprime to all others -> all other sums must be odd
    # So 1+u, 1+v, x+1, x+u, x+v, y+1, y+u, y+v must all be odd
    # 1+u odd -> u even
    # 1+v odd -> v even  
    # x+1 odd -> x even
    # x+u: even+even = even -> NOT coprime to 2!
    
    # So if 2 is in the sumset, we can't have any other even sum.
    # This means if a1=1 and b1=1 (sum=2), then all other sums must be odd.
    # But x+u where x,u both even gives even sum. Contradiction.
    
    # So we can't have both a1=1 and b1=1 if we want x,u even.
    # Let's avoid having 2 in the sumset.
    
    # Try A = {a, a+d1, a+d1+d2} and B = {b, b+e1, b+e1+e2}
    # with a+b >= 3 to avoid small factors
    
    # Let's just do a smart random search with the coprime pruning
    
    start_time = time.time()
    deadline = start_time + 50
    
    # Precompute: for each pair of differences (d1,d2) and (e1,e2), 
    # check if |d1|,|d2|,|d1+d2|,|e1|,|e2|,|e1+e2| are all distinct
    compatible_diffs = {}
    for d1 in range(1, 128):
        for d2 in range(1, 128 - d1 + 1):
            d3 = d1 + d2
            diffs_A = {d1, d2, d3}
            for e1 in range(1, 128):
                for e2 in range(1, 128 - e1 + 1):
                    e3 = e1 + e2
                    diffs_B = {e1, e2, e3}
                    if diffs_A.isdisjoint(diffs_B):
                        if (d1, d2) not in compatible_diffs:
                            compatible_diffs[(d1, d2)] = []
                        compatible_diffs[(d1, d2)].append((e1, e2))
    
    # Now for each compatible pair, try to find actual values
    # A = {a, a+d1, a+d1+d2}, B = {b, b+e1, b+e1+e2}
    # Sums:
    #   a+b, a+b+e1, a+b+e1+e2
    #   a+d1+b, a+d1+b+e1, a+d1+b+e1+e2
    #   a+d1+d2+b, a+d1+d2+b+e1, a+d1+d2+b+e1+e2
    
    # Let s = a+b. Then sums are:
    #   s, s+e1, s+e1+e2
    #   s+d1, s+d1+e1, s+d1+e1+e2
    #   s+d1+d2, s+d1+d2+e1, s+d1+d2+e1+e2
    
    # All must be in [2, 256] and pairwise coprime.
    # s >= 2, s+d1+d2+e1+e2 <= 256
    
    # Let's iterate over compatible diff pairs and try small s values
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    # Check the initial best
    score, valid = _fast_evaluate(best_A, best_B, {})
    if valid:
        best_score = score
    
    # Iterate over compatible difference structures
    items = list(compatible_diffs.items())
    rng.shuffle(items)
    
    for (d1, d2), e_pairs in items:
        if time.time() > deadline:
            break
        d3 = d1 + d2
        
        for (e1, e2) in e_pairs:
            if time.time() > deadline:
                break
            e3 = e1 + e2
            
            # s must satisfy: s >= 2 and s + d3 + e3 <= 256
            # So s in [2, 256 - d3 - e3]
            max_s = 256 - d3 - e3
            if max_s < 2:
                continue
            
            # Try a few values of s
            # s = a + b, where a >= 1, b >= 1, so s >= 2
            # Also a = s - b, and a+d1+d2 <= 128, b+e1+e2 <= 128
            # So s - b + d3 <= 128 -> b >= s + d3 - 128
            # And b + e3 <= 128 -> b <= 128 - e3
            # So b in [max(1, s + d3 - 128), min(128 - e3, s - 1)]
            
            # For each valid b, a = s - b
            # Check if a, a+d1, a+d3 are all in [1, 128] and distinct
            # and b, b+e1, b+e3 are all in [1, 128] and distinct
            
            # Let's try a small number of s values
            s_values = list(range(2, min(max_s + 1, 100)))  # Try small s first
            rng.shuffle(s_values)
            s_values = s_values[:20]  # Limit to 20 per diff pair
            
            for s in s_values:
                if time.time() > deadline:
                    break
                
                # b range
                b_min = max(1, s + d3 - 128)
                b_max = min(128 - e3, s - 1)
                if b_min > b_max:
                    continue
                
                # Try a few b values
                b_candidates = list(range(b_min, b_max + 1))
                if len(b_candidates) > 10:
                    rng.shuffle(b_candidates)
                    b_candidates = b_candidates[:10]
                
                for b in b_candidates:
                    a = s - b
                    if a < 1 or a > 128:
                        continue
                    if a + d3 > 128:
                        continue
                    
                    A = [a, a + d1, a + d3]
                    B = [b, b + e1, b + e3]
                    
                    # Check all in [1, 128] and distinct
                    if any(x < 1 or x > 128 for x in A + B):
                        continue
                    if len(set(A)) != 3 or len(set(B)) != 3:
                        continue
                    
                    # Check if sums are pairwise coprime
                    sums = set()
                    for x in A:
                        for y in B:
                            sums.add(x + y)
                    
                    # Check pairwise coprime
                    sum_list = list(sums)
                    valid = True
                    for i in range(len(sum_list)):
                        for j in range(i + 1, len(sum_list)):
                            if gcd(sum_list[i], sum_list[j]) != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        sc = len(sum_list)
                        if sc > best_score:
                            best_score = sc
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return [best_A, best_B]
    
    # Phase 2: Random search with local improvement (fallback)
    if best_score < 9:
        rng2 = random.Random(42)
        range_ = range(1, 129)
        iteration = 0
        coprime_cache = {}
        
        while time.time() < deadline:
            iteration += 1
            
            # Random set A
            A = list(rng2.sample(range_, 3))
            # Random set B
            B = list(rng2.sample(range_, 3))
            
            score, valid = _fast_evaluate(A, B, coprime_cache)
            if valid and score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]
                if best_score >= 9:
                    return [best_A, best_B]
            
            # Local search: perturb best solution
            if iteration % 3 == 0 and best_score < 9:
                # Try all single-element perturbations of best_A
                for idx in range(3):
                    for new_val in rng2.sample(range_, min(20, 128)):
                        if new_val in best_A and best_A[idx] != new_val:
                            continue
                        A2 = best_A[:]
                        A2[idx] = new_val
                        if len(set(A2)) == 3:
                            score2, valid2 = _fast_evaluate(A2, best_B, coprime_cache)
                            if valid2 and score2 > best_score:
                                best_score = score2
                                best_A = A2[:]
                                if best_score >= 9:
                                    return [best_A, best_B]
                
                # Try all single-element perturbations of best_B
                for idx in range(3):
                    for new_val in rng2.sample(range_, min(20, 128)):
                        if new_val in best_B and best_B[idx] != new_val:
                            continue
                        B2 = best_B[:]
                        B2[idx] = new_val
                        if len(set(B2)) == 3:
                            score3, valid3 = _fast_evaluate(best_A, B2, coprime_cache)
                            if valid3 and score3 > best_score:
                                best_score = score3
                                best_B = B2[:]
                                if best_score >= 9:
                                    return [best_A, best_B]
    
    return [best_A, best_B]

def construct():
    return _backtrack_find_solution(None, 128)
# EVOLVE-BLOCK-END
