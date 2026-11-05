"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_sidon(subset):
    """Check if subset has all unique pair sums (a+b for a<=b)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr):
    """Compute the size of the largest Sidon subset of arr."""
    n = len(arr)
    # Check from largest to smallest
    for size in range(n, 2, -1):
        for subset in combinations(arr, size):
            if _is_sidon(subset):
                return size
    # Check size 3
    for subset in combinations(arr, 3):
        if _is_sidon(subset):
            return 3
    return 2

def _generate_structured_candidates():
    """Generate structured candidates likely to have small max Sidon subsets."""
    candidates = []
    
    # Arithmetic progressions - these have many collisions
    for start in range(0, 12):
        for step in range(1, 11):
            arr = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in arr) and len(set(arr)) == 12:
                candidates.append(tuple(sorted(arr)))
    
    # Block structures: multiple small APs
    for block_size in [3, 4, 6]:
        num_blocks = 12 // block_size
        for gap in range(1, 15):
            for start in range(0, 10):
                arr = []
                for b in range(num_blocks):
                    base = start + b * (block_size + gap)
                    for i in range(block_size):
                        arr.append(base + i)
                if len(arr) == 12 and all(0 <= x <= 128 for x in arr) and len(set(arr)) == 12:
                    candidates.append(tuple(sorted(arr)))
    
    # Modular / quadratic structures
    for a in range(1, 15):
        for b in range(0, 10):
            arr = [(a * i * i + b * i) % 129 for i in range(12)]
            if len(set(arr)) == 12 and all(0 <= x <= 128 for x in arr):
                candidates.append(tuple(sorted(arr)))
    
    # Specific hand-crafted candidates
    candidates.append(tuple(sorted([0,1,2,3,4,5,6,7,8,9,10,11])))
    candidates.append(tuple(sorted([0,1,2,10,11,12,20,21,22,30,31,32])))
    candidates.append(tuple(sorted([0,1,2,3,10,11,12,13,20,21,22,23])))
    candidates.append(tuple(sorted([0,2,4,6,8,10,12,14,16,18,20,22])))
    candidates.append(tuple(sorted([0,1,3,7,12,18,25,33,42,52,63,75])))
    
    return list(set(candidates))

def _local_search(initial, max_time=45):
    """Local search to minimize max Sidon subset size."""
    start_time = time.time()
    current = list(initial)
    current_score = _max_sidon_subset_size(current)
    
    best = list(current)
    best_score = current_score
    
    if best_score <= 3:
        return best, best_score
    
    random.seed(12345)
    
    while time.time() - start_time < max_time:
        improved = False
        
        # Try single-element replacements
        for i in range(12):
            if time.time() - start_time >= max_time:
                break
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
                        if best_score <= 3:
                            return best, best_score
                    improved = True
                    break
                elif score == best_score and random.random() < 0.05:
                    current = new_arr
                    improved = True
                    break
        
        if not improved:
            # Multi-element perturbation
            new_arr = list(current)
            for _ in range(random.randint(2, 4)):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in new_arr or new_val == new_arr[idx]:
                    new_arr[idx] = new_val
            if len(set(new_arr)) == 12:
                score = _max_sidon_subset_size(new_arr)
                if score <= current_score:
                    current = new_arr
                    current_score = score
                    if score < best_score:
                        best = list(current)
                        best_score = score
                        if best_score <= 3:
                            return best, best_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin before 60s limit
    
    # Generate and evaluate structured candidates
    candidates = _generate_structured_candidates()
    
    best_arr = None
    best_score = 12
    
    for arr in candidates:
        if time.time() - start_time > deadline:
            break
        score = _max_sidon_subset_size(list(arr))
        if score < best_score:
            best_score = score
            best_arr = list(arr)
        if best_score <= 3:
            break
    
    # Add random candidates if time permits
    if best_score > 3 and time.time() - start_time < deadline - 10:
        random.seed(42)
        for _ in range(200):
            if time.time() - start_time > deadline:
                break
            arr = random.sample(range(129), 12)
            score = _max_sidon_subset_size(arr)
            if score < best_score:
                best_score = score
                best_arr = list(arr)
            if best_score <= 3:
                break
    
    if best_arr is None:
        best_arr = list(range(12))
        best_score = _max_sidon_subset_size(best_arr)
    
    # Local search from the best candidate
    if time.time() - start_time < deadline - 5:
        remaining = deadline - (time.time() - start_time)
        improved_arr, improved_score = _local_search(best_arr, max_time=min(40, remaining))
        if improved_score < best_score:
            best_arr = improved_arr
            best_score = improved_score
    
    return best_arr

# EVOLVE-BLOCK-END
