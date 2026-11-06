"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if subset is Sidon: all unordered pair sums (i<=j) are unique."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(S):
    """Find the maximum size of a Sidon subset of S via exhaustive search."""
    n = len(S)
    best = 0
    # Try all subsets in decreasing size order
    for size in range(n, 0, -1):
        found = False
        for combo in combinations(range(n), size):
            sub = [S[i] for i in combo]
            if _is_sidon(sub):
                best = size
                found = True
                break
        if found:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    # Start with arithmetic progression
    best_set = list(range(12))
    best_score = _max_sidon_size(best_set)
    
    # Try various structured sets
    candidates = []
    
    # Arithmetic progressions with different steps
    for step in range(1, 11):
        s = list(range(0, 12 * step, step))
        if len(s) == 12 and s[-1] <= 128:
            candidates.append(s)
    
    # Sets with clustering
    # Two clusters
    for base1 in range(0, 50):
        for base2 in range(base1 + 1, 100):
            s = list(range(base1, base1 + 6)) + list(range(base2, base2 + 6))
            if len(set(s)) == 12 and max(s) <= 128:
                candidates.append(s)
    
    # Random sets
    random.seed(42)
    for _ in range(2000):
        s = random.sample(range(129), 12)
        candidates.append(s)
    
    # Evaluate candidates
    for s in candidates:
        if time.time() > deadline:
            break
        score = _max_sidon_size(s)
        if score < best_score:
            best_score = score
            best_set = s
    
    # Local search: try swapping elements
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_set_copy = list(best_set)
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                for new_val in range(129):
                    if new_val in best_set_copy and best_set_copy[i] != new_val:
                        continue
                    old_val = best_set_copy[i]
                    best_set_copy[i] = new_val
                    if len(set(best_set_copy)) != 12:
                        best_set_copy[i] = old_val
                        continue
                    score = _max_sidon_size(best_set_copy)
                    if score < best_score:
                        best_score = score
                        best_set = list(best_set_copy)
                        improved = True
                    best_set_copy[i] = old_val
    
    return best_set
# EVOLVE-BLOCK-END
