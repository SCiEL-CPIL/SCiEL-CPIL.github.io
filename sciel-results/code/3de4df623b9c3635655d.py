"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _is_sidon(subset):
    """Check if a set of integers is Sidon (all pair sums including diagonals are unique)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient_set):
    """Find the largest Sidon subset size using iterative deepening from top down."""
    n = len(ambient_set)
    if n <= 1:
        return n
    
    # Try from largest possible size down
    for size in range(n, 0, -1):
        # Check if any subset of this size is Sidon
        for combo in combinations(range(n), size):
            subset = [ambient_set[i] for i in combo]
            if _is_sidon(subset):
                return size
    return 1

def _max_sidon_size_fast(ambient_set):
    """Faster max Sidon size using greedy + verification."""
    n = len(ambient_set)
    if n <= 1:
        return n
    
    # First check if the whole set is Sidon
    if _is_sidon(ambient_set):
        return n
    
    # Try size n-1
    for i in range(n):
        subset = [ambient_set[j] for j in range(n) if j != i]
        if _is_sidon(subset):
            return n - 1
    
    # Try size n-2
    for i in range(n):
        for j in range(i+1, n):
            subset = [ambient_set[k] for k in range(n) if k != i and k != j]
            if _is_sidon(subset):
                return n - 2
    
    # For smaller sizes, use the full search
    return _max_sidon_size(ambient_set)

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    rng = random.Random(42)
    
    # Generate initial candidate using a structured approach
    # Try to create sets with many sum collisions
    # Strategy: use arithmetic progressions and clustered values
    
    best_set = None
    best_score = 12  # worst case
    
    # Try several structured initializations
    initial_candidates = [
        list(range(12)),           # consecutive
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # AP step 2
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],   # odd + 0
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],     # clustered
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],       # mostly consecutive
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    
    for cand in initial_candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = _max_sidon_size_fast(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
    
    # If we haven't found anything good, use random
    if best_set is None:
        best_set = list(range(12))
        best_score = _max_sidon_size_fast(best_set)
    
    current = list(best_set)
    current_score = best_score
    
    # Local search: modify one element at a time, accept if it improves
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        for idx in range(12):
            if time.time() > deadline:
                break
            for _ in range(20):
                new_val = rng.randint(0, 128)
                if new_val in current:
                    continue
                candidate = list(current)
                candidate[idx] = new_val
                score = _max_sidon_size_fast(candidate)
                if score < current_score:
                    current = candidate
                    current_score = score
                    best_score = score
                    best_set = list(current)
                    improved = True
                    no_improve_count = 0
                    break
                elif score == current_score and rng.random() < 0.2:
                    # Accept equal with some probability to escape local optima
                    current = candidate
                    no_improve_count = 0
                    break
            if time.time() > deadline:
                break
        
        if not improved:
            no_improve_count += 1
            # If stuck, try a bigger perturbation
            if no_improve_count % 10 == 0:
                for _ in range(5):
                    idx = rng.randrange(12)
                    new_val = rng.randint(0, 128)
                    if new_val not in current:
                        current[idx] = new_val
                current_score = _max_sidon_size_fast(current)
                if current_score < best_score:
                    best_score = current_score
                    best_set = list(current)
    
    return best_set
# EVOLVE-BLOCK-END
