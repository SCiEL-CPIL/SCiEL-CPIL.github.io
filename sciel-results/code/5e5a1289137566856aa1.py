"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import itertools
import time

# EVOLVE-BLOCK-START
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(nums[i], nums[j]) != 1:
                return False
    return True

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    best_score = 0
    best_result = [[1, 7, 13], [4, 10, 16]]  # fallback
    
    # Precompute: for each pair (a,b) with a in [1,128], b in [1,128], sum = a+b
    # We need 3x3 grid of sums that are all distinct and pairwise coprime.
    
    # Strategy: enumerate A from small candidates, then for each A find best B.
    # To keep it fast, limit search space.
    
    # Candidate pool: numbers 1..128
    # For speed, try all A (C(128,3) is too large), so use heuristic search.
    
    # Better: random search + local improvement
    import random
    random.seed(42)
    
    def score_pair(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) < 2:
            return 0
        if not _pairwise_coprime(list(sums)):
            return 0
        return len(sums)
    
    # Try structured candidates first
    # Use primes and prime multiples to help coprimality
    primes = [p for p in range(2, 129) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
    
    # Generate candidate sets: mix of small numbers, primes, prime*small
    candidates = list(range(1, 129))
    
    # Phase 1: Try all triples from a reduced candidate set (say first 40 numbers)
    # C(40,3) = 9880, manageable
    reduced = list(range(1, 41))
    
    best_local = 0
    best_A = None
    best_B = None
    
    # For each A (from reduced set), find best B
    for A in itertools.combinations(reduced, 3):
        if time.time() > deadline:
            break
        A = list(A)
        best_B_for_A = None
        best_score_A = 0
        for B in itertools.combinations(reduced, 3):
            B = list(B)
            s = score_pair(A, B)
            if s > best_score_A:
                best_score_A = s
                best_B_for_A = B
        if best_score_A > best_local:
            best_local = best_score_A
            best_A = A
            best_B = best_B_for_A
            if best_local == 9:  # maximum possible
                break
    
    if best_local > best_score:
        best_score = best_local
        best_result = [best_A, best_B]
        if best_score == 9:
            return best_result
    
    # Phase 2: Random search over full range [1,128]
    for _ in range(500000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        s = score_pair(A, B)
        if s > best_score:
            best_score = s
            best_result = [A, B]
            if best_score == 9:
                return best_result
    
    # Phase 3: Local search around best
    if best_score >= 5:
        A0 = list(best_result[0])
        B0 = list(best_result[1])
        for _ in range(200000):
            if time.time() > deadline:
                break
            A = list(A0)
            B = list(B0)
            # mutate one element
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                A[idx] = random.randint(1, 128)
            else:
                idx = random.randint(0, 2)
                B[idx] = random.randint(1, 128)
            # ensure distinctness
            if len(set(A)) != 3 or len(set(B)) != 3:
                continue
            s = score_pair(A, B)
            if s > best_score:
                best_score = s
                best_result = [A, B]
                A0 = A
                B0 = B
                if best_score == 9:
                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
