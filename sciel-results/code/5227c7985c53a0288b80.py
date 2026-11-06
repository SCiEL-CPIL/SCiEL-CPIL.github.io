"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _greedy_max_sidon_size(A):
    """
    Placeholder for compatibility. The exact solver is used in _evaluate_set.
    """
    return 0

def _exact_max_sidon_size(A):
    """
    Exhaustively computes the largest subset with all unordered pair sums unique.
    N is small (12), so this is fast.
    """
    A_sorted = sorted(A)
    n = len(A_sorted)
    
    def is_sidon(subset):
        sums = set()
        m = len(subset)
        for i in range(m):
            for j in range(i, m):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Check subsets from largest size to smallest
    for k in range(n, 0, -1):
        for comb in combinations(A_sorted, k):
            if is_sidon(comb):
                return k
    return 0

def _evaluate_set(A):
    return _exact_max_sidon_size(A)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    candidates = []
    
    # Generate structured candidates likely to have small max Sidon subsets
    # APs with small steps
    for d in range(1, 15):
        if 11 * d <= 128:
            candidates.append(list(range(0, 12 * d, d)))
            
    # Clustered sets
    candidates.append(sorted([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]))
    candidates.append(sorted([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]))
    candidates.append(sorted([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25]))
    
    # Random dense sets (small range)
    random.seed(42)
    for _ in range(50):
        candidates.append(sorted(random.sample(range(25), 12)))
        
    # Random medium density
    for _ in range(50):
        candidates.append(sorted(random.sample(range(50), 12)))

    best_set = None
    best_val = 12
    
    # Evaluate candidates
    for c in candidates:
        if time.time() > deadline:
            break
        val = _evaluate_set(c)
        if val < best_val:
            best_val = val
            best_set = c[:]
        if best_val <= 3:
            break
            
    if best_set is None:
        best_set = list(range(12))
        
    return best_set
# EVOLVE-BLOCK-END
