"""Program to find optimal A, B for pairwise-coprime sumset maximization."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    vs = list(set(vals))
    n = len(vs)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _eval(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) <= 1:
        return len(sums)
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct integers in [1,128]."""
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    return True

def _is_coprime_with_all(val, existing_set):
    """Check if val is coprime with all values in existing_set."""
    for v in existing_set:
        if _gcd(val, v) != 1:
            return False
    return True

def _search():
    """Search for optimal A, B using structured grid approach."""
    deadline = time.time() + 58  # 58s budget
    
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    # Precompute primes up to 256 for fast primality check
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Generate a pool of "candidate sums" that are likely to be pairwise coprime.
    # Primes are the best candidates. We can also include prime powers.
    candidates = []
    for n in range(2, 257):
        if is_prime[n]:
            candidates.append(n)
    
    # We can also add 1, but 1 is coprime to everything.
    # Let's focus on primes first as they are most abundant and structurally simple.
    
    # We want to form a 3x3 grid of sums S_ij = A_i + B_j.
    # Let A = [1, 1+d2, 1+d3] and B = [b1, b1+d_b1, b1+d_b2]? No, let's fix A and search B or vice versa.
    # Simpler: Fix the first row of sums: s11, s12, s13.
    # These must be pairwise coprime.
    # Then s21 = s11 + d2, s22 = s12 + d2, s23 = s13 + d2.
    # And s31 = s11 + d3, s32 = s12 + d3, s33 = s13 + d3.
    # We need all 9 to be distinct and pairwise coprime.
    # Also A and B must be in [1, 128].
    # If we set A1 = 1, then B_j = s1j - 1.
    # A2 = 1 + d2 => B_j + A2 = s1j + d2.
    # A3 = 1 + d3 => B_j + A3 = s1j + d3.
    
    # Iterate over possible first rows (s11, s12, s13) from candidates
    # To keep it fast, we limit the range of s1j.
    # Since B_j = s1j - 1 must be in [1, 128], s1j must be in [2, 129].
    # Let's iterate over s11, s12, s13 in range(2, 130).
    
    # Optimization: Precompute valid triples (x, y, z) that are pairwise coprime.
    # But 128^3 is ~2M, which is fine.
    
    # Even better: Iterate s11, s12, s13.
    # Then iterate d2, d3.
    # Total iterations: 128^3 * 128^2 is too high.
    
    # Let's restrict d2, d3 to small values first, or use a smarter search.
    # We can iterate over A and B directly but with pruning.
    
    # Let's try a different structured search:
    # Fix A = [1, 2, 3]? No, that's too specific.
    # Fix A1 = 1.
    # Iterate B1, B2, B3.
    # Then find A2, A3.
    
    # Let's iterate over B1, B2, B3 in [1, 128].
    # For each B, we want to find A2, A3 such that:
    # 1. A2+B1, A2+B2, A2+B3 are pairwise coprime with each other and with B1, B2, B3 (shifted by A1=1? No, A1 is fixed).
    # Wait, if A1=1, sums are 1+B1, 1+B2, 1+B3.
    # Let R1 = [1+B1, 1+B2, 1+B3].
    # We need R1 to be pairwise coprime.
    # Then for A2, sums are R1 + (A2-1). Let d2 = A2-1.
    # R2 = [r1+d2, r2+d2, r3+d2].
    # R2 must be pairwise coprime and coprime to R1.
    # Similarly for A3.
    
    # Strategy:
    # 1. Generate all valid R1 triples (pairwise coprime, values in [2, 129]).
    # 2. For each R1, find all valid d2 such that R1+d2 is valid (pairwise coprime, coprime to R1).
    # 3. For each R1 and valid d2, find all valid d3 such that R1+d3 is valid (pairwise coprime, coprime to R1 and R1+d2).
    # 4. If found, construct A and B and check bounds.
    
    valid_R1s = []
    for r1 in range(2, 130):
        for r2 in range(r1+1, 130):
            if _gcd(r1, r2) != 1: continue
            for r3 in range(r2+1, 130):
                if _gcd(r1, r3) != 1 or _gcd(r2, r3) != 1: continue
                valid_R1s.append((r1, r2, r3))
                
    # This list might be large, but 128^3/6 is ~350k.
    # For each R1, we check d2 in [1, 127] (since A2 <= 128, A1=1 => d2 <= 127).
    # And d3 in [1, 127].
    # This is 350k * 128 * 128 ~ 5.7 billion operations. Too slow.
    
    # We need to prune more.
    # Note that R1 values are small.
    # Let's just iterate d2, d3 first?
    
    # Alternative: Random search with a better heuristic.
    # The previous random search was weak.
    # Let's use a "greedy" local search.
    # Start with a random valid A, B.
    # Try to swap elements or shift them to increase the score.
    
    rng = random.Random(42)
    
    # Initial population
    best_A = None
    best_B = None
    best_score = 0
    
    # Try to find a good starting point
    for _ in range(10000):
        A = list(rng.sample(range(1, 129), 3))
        B = list(rng.sample(range(1, 129), 3))
        s = _eval(A, B)
        if s > best_score:
            best_score = s
            best_A = A
            best_B = B
            if best_score == 9:
                return best_A, best_B

    if best_A is None:
        return [1, 2, 3], [1, 2, 3]
        
    A = best_A[:]
    B = best_B[:]
    
    # Local search
    # We can try to modify A or B.
    # Since the space is small, we can try all single-element changes.
    # If we change A[0] to x, we need to check if the new set is valid.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try changing each element of A
        for i in range(3):
            old_val = A[i]
            for new_val in range(1, 129):
                if new_val == old_val: continue
                if new_val in A: continue
                A_new = A[:]
                A_new[i] = new_val
                s = _eval(A_new, B)
                if s > best_score:
                    best_score = s
                    best_A = A_new
                    best_B = B[:]
                    A = A_new
                    improved = True
                    if best_score == 9:
                        return best_A, best_B
                # If s is equal, we might still want to explore? No, we want to maximize.
                # But maybe a plateau leads to a better peak?
                # For now, only accept improvements.
                
        # Try changing each element of B
        for i in range(3):
            old_val = B[i]
            for new_val in range(1, 129):
                if new_val == old_val: continue
                if new_val in B: continue
                B_new = B[:]
                B_new[i] = new_val
                s = _eval(A, B_new)
                if s > best_score:
                    best_score = s
                    best_A = A[:]
                    best_B = B_new
                    B = B_new
                    improved = True
                    if best_score == 9:
                        return best_A, best_B
                        
    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END
