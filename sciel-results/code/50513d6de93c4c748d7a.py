"""Search-based program: find (A,B) with 3 distinct sums, pairwise coprime, maximizing |A+B|."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

def _gcd_tuple_ok(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    lst = list(vals)
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(lst[i], lst[j]) != 1:
                return False
    return True


def _score(A, B):
    """Return the number of distinct sums if valid, else -1."""
    s = {a + b for a in A for b in B}
    if len(s) < 2:
        return -1
    if not _gcd_tuple_ok(s):
        return -1
    return len(s)


def construct():
    deadline = time.time() + 80  # safety margin under 90s wall-clock
    best_score = 0
    best_result = [[1, 7, 13], [4, 10, 16]]  # seed
    
    # Verify seed
    seed_score = _score(best_result[0], best_result[1])
    if seed_score > best_score:
        best_score = seed_score

    rng = random.Random(42)
    range_n = 128

    # Strategy 1: Exhaustive search over small ranges first
    # Try all combinations of 3 elements from [1..N] for A and B
    # For small N this is feasible
    
    for N in range(20, 61):
        if time.time() > deadline:
            break
        # Precompute all 3-element subsets from [1..N]
        subsets = list(combinations(range(1, N + 1), 3))
        num_subsets = len(subsets)
        
        for i in range(num_subsets):
            if time.time() > deadline:
                break
            A = list(subsets[i])
            for j in range(i, num_subsets):  # symmetry: B can equal A
                B = list(subsets[j])
                s = _score(A, B)
                if s > best_score:
                    best_score = s
                    best_result = [A, B]

    # Strategy 2: Random search with larger range
    while time.time() < deadline:
        A = sorted(rng.sample(range(1, range_n + 1), 3))
        B = sorted(rng.sample(range(1, range_n + 1), 3))
        s = _score(A, B)
        if s > best_score:
            best_score = s
            best_result = [A, B]

    return best_result

# EVOLVE-BLOCK-END
