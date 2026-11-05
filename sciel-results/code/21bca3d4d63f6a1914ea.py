"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _count_duplicate_sums(arr):
    """Count the number of duplicate unordered pair sums (including diagonals).
    Higher means more collisions, which is good for minimizing Sidon subset size."""
    sums = {}
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums[s] = sums.get(s, 0) + 1
    # Count duplicates: for each sum value with count c, it contributes c-1 to the "excess"
    # Total pairs = n*(n+1)/2. Unique sums = len(sums). Duplicates = total_pairs - len(sums)
    total_pairs = n * (n + 1) // 2
    return total_pairs - len(sums)

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    best = 0
    # Try subsets in decreasing size order
    for size in range(n, best, -1):
        for subset in combinations(arr, size):
            # Check if this subset is Sidon
            sums = set()
            is_sidon = True
            for i in range(size):
                for j in range(i, size):
                    s = subset[i] + subset[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                return size
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_arr = list(range(12))
    best_collisions = _count_duplicate_sums(best_arr)
    
    # We want to MAXIMIZE collisions to minimize the largest Sidon subset
    # Start with random initializations and greedy swaps
    
    random.seed(42)
    
    for init_iter in range(50):
        if time.time() > deadline:
            break
        
        # Random initial 12-element subset from [0, 128]
        arr = sorted(random.sample(range(129), 12))
        collisions = _count_duplicate_sums(arr)
        
        # Greedy local search: try swapping each element with each other value
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                for new_val in range(129):
                    if new_val in arr:
                        continue
                    # Try replacing arr[i] with new_val
                    new_arr = arr.copy()
                    new_arr[i] = new_val
                    new_arr = sorted(new_arr)
                    new_collisions = _count_duplicate_sums(new_arr)
                    if new_collisions > collisions:
                        arr = new_arr
                        collisions = new_collisions
                        improved = True
                        break
        
        if collisions > best_collisions:
            best_arr = arr
            best_collisions = collisions
    
    # Verify the result
    return best_arr
# EVOLVE-BLOCK-END
