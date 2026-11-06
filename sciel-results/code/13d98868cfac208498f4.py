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

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # known valid score
    
    rng = random.Random(42)
    coprime_cache = {}
    
    # Phase 1: Try structured candidates
    # Sums should be pairwise coprime. If we can get 9 distinct sums, that's max.
    # For 9 distinct sums from 3x3, all a_i + b_j must be distinct.
    # This requires A and B to be in "general position" (no a_i - a_j = b_k - b_l for i!=j, k!=l)
    
    # Try sets where differences are all distinct
    # A = {a1, a2, a3}, B = {b1, b2, b3}
    # Sums: a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    # For all 9 to be distinct: a_i - a_j != b_k - b_l for all i≠j, k≠l (with sign consideration)
    # Actually: a_i + b_j = a_k + b_l iff a_i - a_k = b_l - b_j
    # So we need all differences in A to be distinct from all differences in B (up to sign)
    
    # Try small structured sets
    test_candidates = [
        ([1, 2, 4], [1, 3, 8]),
        ([1, 2, 4], [1, 3, 9]),
        ([1, 2, 5], [1, 3, 7]),
        ([1, 3, 5], [1, 4, 7]),
        ([1, 2, 6], [1, 3, 8]),
        ([1, 3, 6], [1, 4, 7]),
        ([2, 3, 7], [1, 4, 8]),
        ([1, 4, 6], [1, 3, 8]),
        ([1, 2, 5], [1, 4, 7]),
        ([1, 3, 7], [1, 2, 5]),
    ]
    
    for A_try, B_try in test_candidates:
        score, valid = _fast_evaluate(A_try, B_try, coprime_cache)
        if valid and score > best_score:
            best_score = score
            best_A = A_try[:]
            best_B = B_try[:]
            if best_score >= 9:
                return [best_A, best_B]
    
    # Phase 2: Random search with local improvement
    range_ = range(1, 129)
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Random set A
        A = list(rng.sample(range_, 3))
        # Random set B
        B = list(rng.sample(range_, 3))
        
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
                for new_val in rng.sample(range_, min(20, 128)):
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
                for new_val in rng.sample(range_, min(20, 128)):
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
# EVOLVE-BLOCK-END
