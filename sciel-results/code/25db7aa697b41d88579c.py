import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a subset has all pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_subset_size(arr):
    """Find the largest subset of arr that is Sidon."""
    n = len(arr)
    best = 0
    # Try subsets from largest to smallest
    for size in range(n, 0, -1):
        for subset in combinations(arr, size):
            if is_sidon(subset):
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = 80.0  # 80s safety margin before 90s kill
    
    # Start with a random set
    best_set = list(range(12))
    best_score = max_sidon_subset_size(best_set)
    
    random.seed(42)
    
    current_set = list(range(12))
    current_score = max_sidon_subset_size(current_set)
    best_set = current_set[:]
    best_score = current_score
    
    # Try random initializations
    while time.time() - start_time < deadline:
        # Generate a random set of 12 distinct integers in [0, 128]
        candidate = sorted(random.sample(range(129), 12))
        score = max_sidon_subset_size(candidate)
        
        if score < best_score:
            best_score = score
            best_set = candidate[:]
            current_set = candidate[:]
            current_score = score
            
            if best_score <= 5:
                break
        
        # Local search: try swapping one element
        if time.time() - start_time > deadline:
            break
            
        for _ in range(100):
            if time.time() - start_time > deadline:
                break
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current_set:
                continue
            old_val = current_set[i]
            current_set[i] = new_val
            current_set.sort()
            
            new_score = max_sidon_subset_size(current_set)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_set[:]
                    if best_score <= 5:
                        break
            else:
                # Revert
                current_set[i] = old_val
                current_set.sort()
    
    return best_set
