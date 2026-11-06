"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if a set of integers is Sidon (all pair sums i<=j distinct)."""
    s = sorted(subset)
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _max_sidon_size(arr):
    """Find the largest Sidon subset size of arr (size <= 12)."""
    n = len(arr)
    # Try from largest to smallest
    for k in range(n, 2, -1):
        for combo in combinations(arr, k):
            if _is_sidon(combo):
                return k
    # Check size 3
    for combo in combinations(arr, 3):
        if _is_sidon(combo):
            return 3
    return 2

def construct():
    deadline = time.time() + 80.0  # safety margin
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    # Try some structured sets first
    candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15],
    ]
    
    for cand in candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        sz = _max_sidon_size(cand)
        if sz < best_score:
            best_score = sz
            best_set = cand[:]
    
    # Local search: try to improve by swapping elements
    random.seed(42)
    current = best_set[:]
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        # Try random swaps
        for _ in range(200):
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            sz = _max_sidon_size(current)
            if sz < current_score:
                current_score = sz
                improved = True
                if sz <= 3:
                    current[i] = old_val
                    current[i] = new_val
                    break
            else:
                current[i] = old_val
        
        if current_score <= 3:
            break
        if not improved:
            # Try a few more random restarts
            for _ in range(5):
                if time.time() > deadline:
                    break
                current = [random.randint(0, 128) for _ in range(12)]
                if len(set(current)) != 12:
                    continue
                sz = _max_sidon_size(current)
                if sz < current_score:
                    current_score = sz
                    improved = True
            if not improved:
                break
    
    if current_score < best_score:
        best_set = current[:]
        best_score = current_score
    
    return best_set
# EVOLVE-BLOCK-END
