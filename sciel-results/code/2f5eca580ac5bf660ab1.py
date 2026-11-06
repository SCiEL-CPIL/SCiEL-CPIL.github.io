"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

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

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s kill
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # known valid score
    
    # Try to improve via local search / random restarts
    rng = random.Random(42)
    
    # Precompute: for quick coprime checks, we can use the fact that
    # if all sums are primes, they're pairwise coprime.
    # But also composites can be pairwise coprime (e.g., 4 and 9).
    
    # Strategy: random search over triples
    range_ = range(1, 129)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Random set A
        A = list(rng.sample(range_, 3))
        # Random set B
        B = list(rng.sample(range_, 3))
        
        score, valid = _evaluate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                break  # maximum possible
        
        # Also try local perturbation of best
        if iteration % 5 == 0 and best_score < 9:
            # Perturb one element of best_A
            A2 = best_A[:]
            idx = rng.randint(0, 2)
            A2[idx] = rng.randint(1, 128)
            if len(set(A2)) == 3:
                score2, valid2 = _evaluate(A2, best_B)
                if valid2 and score2 > best_score:
                    best_score = score2
                    best_A = A2[:]
                    if best_score >= 9:
                        break
            
            # Perturb one element of best_B
            B2 = best_B[:]
            idx = rng.randint(0, 2)
            B2[idx] = rng.randint(1, 128)
            if len(set(B2)) == 3:
                score3, valid3 = _evaluate(best_A, B2)
                if valid3 and score3 > best_score:
                    best_score = score3
                    best_B = B2[:]
                    if best_score >= 9:
                        break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
