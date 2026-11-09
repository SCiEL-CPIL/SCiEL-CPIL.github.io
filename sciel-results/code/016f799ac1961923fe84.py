"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if subset has all unordered pair sums unique (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(arr):
    """Find the largest Sidon subset size of arr via exhaustive search with pruning."""
    n = len(arr)
    best = 0
    # Try subsets in decreasing size
    for size in range(n, 0, -1):
        for combo in combinations(range(n), size):
            if _is_sidon([arr[i] for i in combo]):
                return size
    return 0

def _score(arr):
    """Lower is better: the max Sidon subset size."""
    return _max_sidon_size(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit
    
    # Initial structured guess: blocks with internal collisions
    # {0,1,2, 4,5,6, 8,9,10, 12,13,14}
    best = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    best_score = _score(best)
    
    # Try a few hand-crafted structured sets
    candidates = [
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15],
        [0, 1, 2, 4, 5, 8, 9, 10, 12, 13, 16, 17],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 17],
        [0, 1, 2, 4, 6, 7, 8, 10, 12, 13, 14, 16],
    ]
    
    for cand in candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            s = _score(cand)
            if s < best_score:
                best_score = s
                best = list(cand)
    
    # Local search: swap elements to improve
    random.seed(42)
    current = list(best)
    current_score = best_score
    T = 0.5  # temperature for simulated annealing
    
    while time.time() < deadline:
        # Pick two random positions to swap or mutate
        i, j = random.sample(range(12), 2)
        # Mutate: replace one element with a random value
        old_val = current[i]
        new_val = random.randint(0, 128)
        if new_val in current:
            continue
        current[i] = new_val
        
        new_score = _score(current)
        
        if new_score < current_score:
            best = list(current)
            best_score = new_score
            current_score = new_score
            T = max(0.1, T * 0.95)
        else:
            # Accept worse with probability
            delta = new_score - current_score
            if random.random() < (1.0 / (1.0 + delta * 2)):
                current_score = new_score
            else:
                current[i] = old_val
        T = max(0.05, T * 0.999)
        
        # Occasional restart if stuck
        if random.random() < 0.01:
            current = list(best)
            current_score = best_score
    
    # Ensure validity
    if len(set(best)) != 12 or not all(0 <= x <= 128 for x in best):
        return list(range(12))
    
    return best
# EVOLVE-BLOCK-END
