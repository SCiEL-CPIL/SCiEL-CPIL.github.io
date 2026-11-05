"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _max_sidon_subset_size(arr):
    """Compute the size of the largest subset of arr where all pair sums (a+b for a<=b) are unique.
    Uses backtracking with pruning."""
    n = len(arr)
    arr_sorted = sorted(arr)
    
    # Quick upper bound: max Sidon subset size
    # For a Sidon set of size k, we need k*(k+1)/2 unique sums
    # The sums range from 2*min to 2*max, so at most 2*(max-min)+1 distinct values
    max_possible = 12
    for size in range(n, 0, -1):
        if _exists_sidon_subset(arr_sorted, size):
            return size
    return 0

def _exists_sidon_subset(arr_sorted, target_size):
    """Backtracking to check if a Sidon subset of target_size exists."""
    n = len(arr_sorted)
    if target_size <= 2:
        return True
    
    def backtrack(start, chosen, seen_sums):
        if len(chosen) == target_size:
            return True
        remaining = target_size - len(chosen)
        if n - start < remaining:
            return False
        
        for i in range(start, n):
            x = arr_sorted[i]
            # Check all new pair sums
            ok = True
            new_sums = []
            for j in range(len(chosen)):
                s = chosen[j] + x
                if s in seen_sums:
                    ok = False
                    break
                new_sums.append(s)
            if not ok:
                continue
            s_diag = 2 * x
            if s_diag in seen_sums:
                continue
            new_sums.append(s_diag)
            
            chosen.append(x)
            seen_sums.update(new_sums)
            
            if backtrack(i + 1, chosen, seen_sums):
                return True
            
            chosen.pop()
            seen_sums.difference_update(new_sums)
        
        return False
    
    return backtrack(0, [], set())

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
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(12):
            for _ in range(30):
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
                    current = new_arr
                    improved = True
                    break
        if not improved:
            for _ in range(5):
                if time.time() - start_time > max_time:
                    break
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
    deadline = 80  # Safety margin before 90s kill
    
    candidates = []
    
    # Arithmetic progressions - many additive relations
    for start in range(0, 15):
        for step in range(1, 15):
            arr = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in arr) and len(set(arr)) == 12:
                candidates.append(arr)
    
    # Quadratic-like structures
    for base in range(0, 20):
        for offset in range(0, 10):
            arr = [base + offset + i * (i + 1) // 2 for i in range(12)]
            if all(0 <= x <= 128 for x in arr) and len(set(arr)) == 12:
                candidates.append(arr)
    
    # Clustered sets
    for center in range(5, 124, 3):
        for spread in range(3, 12):
            lo = max(0, center - spread)
            hi = min(128, center + spread)
            if hi - lo + 1 >= 12:
                random.seed(center * 100 + spread)
                arr = random.sample(range(lo, hi + 1), 12)
                candidates.append(arr)
    
    # Random candidates
    random.seed(42)
    for _ in range(200):
        arr = random.sample(range(129), 12)
        candidates.append(arr)
    
    best_arr = None
    best_score = 12
    
    for arr in candidates:
        if time.time() - start_time > deadline:
            break
        score = _max_sidon_subset_size(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
        if best_score <= 3:
            break
    
    if best_arr is None:
        best_arr = list(range(12))
        best_score = _max_sidon_subset_size(best_arr)
    
    if time.time() - start_time < deadline:
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:
            improved_arr, improved_score = _local_search(best_arr, max_time=min(50, remaining - 2))
            if improved_score < best_score:
                best_arr = improved_arr
                best_score = improved_score
    
    return best_arr
# EVOLVE-BLOCK-END
