"""Rewritten program: structured search for A,B maximizing |A+B| with pairwise-coprime distinct sums."""
import time
import math
import random
import itertools
from functools import lru_cache

_DEADLINE = 85.0  # safety margin before 90s kill

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

def _pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    vals = list(set(values))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if math.gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _sumset(A, B):
    return set(a + b for a in A for b in B)

def _valid_solution(A, B):
    """Check if A and B form a valid solution, return (score, valid)."""
    s = _sumset(A, B)
    if _pairwise_coprime(s):
        return len(s), True
    return len(s), False

def _find_solution_structured():
    """
    Structured search using the key insight:
    - For A+B to have 9 distinct elements, all 9 sums must be distinct.
    - The 9 sums must be pairwise coprime.
    
    Strategy:
    1. Enumerate candidate B sets (small, structured).
    2. For each B, search for A that gives 9 distinct pairwise-coprime sums.
    3. Use backtracking with pruning.
    """
    start_time = time.time()
    rng = random.Random(12345)
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Verify seed
    s_seed = _sumset(best_A, best_B)
    if _pairwise_coprime(s_seed):
        best_score = len(s_seed)
    
    # Precompute: for each pair (a,b), the sum s = a+b and whether s is "good"
    # (i.e., can be part of a pairwise-coprime set)
    
    # Key insight: if we want 9 pairwise-coprime sums, they should ideally be
    # primes or products of distinct primes.
    
    # Let's try a more targeted approach:
    # Fix B to be small, well-separated values.
    # For each B, try A values that create distinct sums.
    
    # Candidate B sets: try small sets with good separation
    B_candidates = []
    
    # Small B sets
    for b1 in range(1, 20):
        for b2 in range(b1 + 2, 30):
            for b3 in range(b2 + 2, 50):
                B_candidates.append([b1, b2, b3])
    
    # Also try larger B sets
    for _ in range(500):
        B_candidates.append(sorted(rng.sample(range(1, 129), 3)))
    
    # For each B, search for A
    for B in B_candidates:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        b1, b2, b3 = B
        
        # For A = [a1, a2, a3], the 9 sums are:
        # a1+b1, a1+b2, a1+b3
        # a2+b1, a2+b2, a2+b3
        # a3+b1, a3+b2, a3+b3
        
        # For 9 distinct sums, we need:
        # a_i + b_j != a_k + b_l for (i,j) != (k,l)
        # This means a_i - a_k != b_l - b_j for all i!=k, j!=l
        
        # Let's try A values in a structured way
        for a1 in range(1, 60):
            if time.time() - start_time > _DEADLINE - 10:
                break
            for a2 in range(a1 + 2, 80):
                if time.time() - start_time > _DEADLINE - 10:
                    break
                for a3 in range(a2 + 2, 100):
                    if time.time() - start_time > _DEADLINE - 5:
                        break
                    
                    A = [a1, a2, a3]
                    s = _sumset(A, B)
                    
                    # Quick check: 9 distinct sums?
                    if len(s) != 9:
                        continue
                    
                    # Check pairwise coprime
                    if _pairwise_coprime(s):
                        if 9 > best_score:
                            best_score = 9
                            best_A, best_B = A, B
                            return best_A, best_B
                        elif 9 == best_score:
                            # Found another 9-element solution, return it
                            best_A, best_B = A, B
                            return best_A, best_B
    
    # If structured search didn't find 9, do a broader random search
    if best_score < 9:
        iterations = 0
        while time.time() - start_time < _DEADLINE - 5:
            A = sorted(rng.sample(range(1, 129), 3))
            B = sorted(rng.sample(range(1, 129), 3))
            s = _sumset(A, B)
            if len(s) == 9 and _pairwise_coprime(s):
                return A, B
            if _pairwise_coprime(s) and len(s) > best_score:
                best_score = len(s)
                best_A, best_B = A, B
            iterations += 1
            if iterations % 100000 == 0:
                break
    
    return best_A, best_B

def _find_solution_exhaustive_small():
    """
    Exhaustive search over small A and B values.
    This is guaranteed to find a solution if one exists in the small range.
    """
    start_time = time.time()
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Verify seed
    s_seed = _sumset(best_A, best_B)
    if _pairwise_coprime(s_seed):
        best_score = len(s_seed)
    
    # Search over A and B in [1, 50]
    # Number of triples: C(50,3) = 19600, so 19600^2 = ~384 million - too many
    # Instead, fix B and search A, or use symmetry
    
    # Let's fix B to small values and search A
    for b1 in range(1, 25):
        if time.time() - start_time > _DEADLINE - 15:
            break
        for b2 in range(b1 + 1, 40):
            if time.time() - start_time > _DEADLINE - 15:
                break
            for b3 in range(b2 + 1, 60):
                if time.time() - start_time > _DEADLINE - 10:
                    break
                
                B = [b1, b2, b3]
                
                for a1 in range(1, 30):
                    if time.time() - start_time > _DEADLINE - 10:
                        break
                    for a2 in range(a1 + 1, 50):
                        if time.time() - start_time > _DEADLINE - 10:
                            break
                        for a3 in range(a2 + 1, 70):
                            if time.time() - start_time > _DEADLINE - 5:
                                break
                            
                            A = [a1, a2, a3]
                            s = _sumset(A, B)
                            
                            if len(s) == 9 and _pairwise_coprime(s):
                                return A, B
                            elif _pairwise_coprime(s) and len(s) > best_score:
                                best_score = len(s)
                                best_A, best_B = A, B
                                if best_score == 9:
                                    return best_A, best_B
    
    return best_A, best_B

def construct():
    """Return two lists A,B each with exactly 3 distinct integers in [1,128].
    All distinct values in A+B must be pairwise coprime. Maximize |A+B|."""
    # Try structured search first
    A, B = _find_solution_structured()
    
    # Verify validity
    s = _sumset(A, B)
    if _pairwise_coprime(s):
        return A, B
    
    # Fallback to known valid solution
    return [1, 7, 13], [4, 10, 16]
