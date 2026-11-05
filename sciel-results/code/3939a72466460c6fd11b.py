"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _max_sidon_subset_size(arr):
    """Compute the size of the largest subset of arr where all pair sums (a+b for a<=b) are unique."""
    n = len(arr)
    best = 0
    # Try all subsets by size, from largest to smallest
    for size in range(n, 1, -1):
        found = False
        for subset in combinations(arr, size):
            s = set(subset)
            pair_sums = set()
            valid = True
            for i, a in enumerate(subset):
                for b in subset[i:]:
                    ps = a + b
                    if ps in pair_sums:
                        valid = False
                        break
                    pair_sums.add(ps)
                if not valid:
                    break
            if valid:
                best = size
                found = True
                break
        if found:
            break
    return best

def _try_candidate(arr):
    """Validate and evaluate a candidate."""
    if len(arr) != 12:
        return None
    if len(set(arr)) != 12:
        return None
    if any(x < 0 or x > 128 for x in arr):
        return None
    return _max_sidon_subset_size(arr)

def _local_search(initial, max_time=50):
    """Local search to minimize max Sidon subset size."""
    start_time = time.time()
    current = list(initial)
    current_score = _max_sidon_subset_size(current)
    
    best = list(current)
    best_score = current_score
    
    # Try single-element perturbations
    while time.time() - start_time < max_time:
        improved = False
        for i in range(12):
            # Try replacing element i with a random value
            for _ in range(20):
                new_val = random.randint(0, 128)
                if new_val in current and new_val != current[i]:
                    continue
                new_arr = list(current)
                new_arr[i] = new_val
                if len(set(new_arr)) != 12:
                    continue
                score = _max_sidon_subset_size(new_arr)
                if score < current_score:
                    current = new_arr
                    current_score = score
                    if score < best_score:
                        best = list(current)
                        best_score = score
                    improved = True
                    break
                elif score == best_score and random.random() < 0.1:
                    # Occasional neutral move to escape local optima
                    current = new_arr
                    improved = True
                    break
        if not improved:
            # Try a random restart with small perturbation
            new_arr = list(current)
            for _ in range(3):
                idx = random.randint(0, 11)
                new_arr[idx] = random.randint(0, 128)
            if len(set(new_arr)) == 12:
                score = _max_sidon_subset_size(new_arr)
                if score <= current_score:
                    current = new_arr
                    current_score = score
                    if score < best_score:
                        best = list(current)
                        best_score = score
            if time.time() - start_time > max_time:
                break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin
    
    # Try several initial candidates
    candidates = []
    
    # Arithmetic progressions
    for start in range(0, 10):
        for step in range(1, 12):
            arr = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in arr) and len(set(arr)) == 12:
                candidates.append(arr)
    
    # Random candidates
    random.seed(42)
    for _ in range(50):
        arr = random.sample(range(129), 12)
        candidates.append(arr)
    
    # Evaluate all candidates quickly
    best_arr = None
    best_score = 12
    
    for arr in candidates:
        if time.time() - start_time > deadline:
            break
        score = _max_sidon_subset_size(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
        if best_score <= 2:
            break
    
    if best_arr is None:
        best_arr = list(range(12))
        best_score = _max_sidon_subset_size(best_arr)
    
    # Local search from the best candidate
    if time.time() - start_time < deadline:
        improved_arr, improved_score = _local_search(best_arr, max_time=min(40, deadline - (time.time() - start_time)))
        if improved_score < best_score:
            best_arr = improved_arr
            best_score = improved_score
    
    return best_arr
# EVOLVE-BLOCK-END
