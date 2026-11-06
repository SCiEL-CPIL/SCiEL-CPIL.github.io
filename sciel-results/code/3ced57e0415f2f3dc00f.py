import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if a subset is a Sidon set (all pair sums i<=j are unique)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(ambient):
    """
    Exhaustively compute the largest subset of ambient with all unordered pair sums unique,
    INCLUDING diagonals.
    """
    n = len(ambient)
    if n <= 0:
        return 0
    # Check from largest k down to 1
    for k in range(n, 0, -1):
        for subset in combinations(ambient, k):
            if _is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Initial candidate: Arithmetic Progression (known to have small Sidon subsets)
    # AP of length 12: 0, 1, ..., 11
    # Let's also try other structures
    initial_candidates = [
        list(range(12)),
        list(range(0, 120, 10)), # 0, 10, ..., 110
        [1, 2, 3, 4, 5, 6, 58, 59, 60, 61, 62, 63], # Pairs summing to 64
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 3, 7, 15, 31, 63, 127, 126, 125, 124, 123], # Mix
    ]
    
    best_set = None
    best_size = 12
    
    # Evaluate initial candidates
    for cand in initial_candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        size = _max_sidon_subset_size(cand)
        if size < best_size:
            best_size = size
            best_set = list(cand)
            
    # If we found size 4, it's likely optimal (theoretical lower bound for N=12 is not strictly 4, but 4 is very small)
    # Let's do a local search to see if we can improve or confirm.
    
    current_set = best_set if best_set else list(range(12))
    current_size = best_size if best_set else 12
    best_global_set = list(current_set)
    best_global_size = current_size
    
    # Local Search / Simulated Annealing
    # We want to minimize the max Sidon size.
    # Objective: size.
    
    # To make search effective, we should focus on sets that are "additively dense".
    # Swapping elements to keep them close together or in structured positions.
    
    rng = random.Random(42)
    
    # Track best
    for iteration in range(5000):
        if time.time() > deadline:
            break
            
        # Generate a neighbor
        # Strategy: Pick a random index and a random new value
        # Or, pick a random index and swap with another random index (preserves sum? no, just changes set)
        # Let's try random replacement.
        
        i = rng.randint(0, 11)
        new_val = rng.randint(0, 128)
        
        if new_val in current_set:
            continue
            
        candidate = list(current_set)
        candidate[i] = new_val
        candidate.sort()
        
        # Evaluate
        # Optimization: If we just changed one element, we can check if the max size decreased.
        # But _max_sidon_subset_size is fast enough for 12 elements.
        
        new_size = _max_sidon_subset_size(candidate)
        
        # Acceptance Criterion
        # Always accept if better.
        # Accept if equal with some probability (to explore).
        # Reject if worse.
        
        if new_size < current_size:
            current_set = candidate
            current_size = new_size
            if new_size < best_global_size:
                best_global_size = new_size
                best_global_set = list(current_set)
                # If we hit 4, we can stop early? 
                # Theoretical min for N=12 might be 4. If we find 4, it's likely the global min.
                # Let's keep searching a bit more to be sure, or stop if 4 is found and time is tight.
        elif new_size == current_size:
            # Accept with probability 0.5 to avoid getting stuck
            if rng.random() < 0.5:
                current_set = candidate
        # else: reject
        
    return best_global_set
