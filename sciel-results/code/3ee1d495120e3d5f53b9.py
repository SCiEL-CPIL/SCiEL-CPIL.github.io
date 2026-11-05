"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _largest_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    # Try subsets in decreasing size order
    for size in range(n, 0, -1):
        found = False
        for subset in combinations(range(n), size):
            sums = set()
            is_sidon = True
            for i in range(size):
                for j in range(i, size):
                    s = arr[subset[i]] + arr[subset[j]]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                return size
    return 1

def _count_duplicate_sums(arr):
    """Count duplicate unordered pair sums. Higher = more collisions."""
    sums = {}
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums[s] = sums.get(s, 0) + 1
    total_pairs = n * (n + 1) // 2
    return total_pairs - len(sums)

def _evaluate(arr):
    """Evaluate a candidate: return (largest_sidon_size, collision_count)."""
    ls_size = _largest_sidon_subset_size(arr)
    coll = _count_duplicate_sums(arr)
    return ls_size, coll

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best_arr = list(range(12))
    best_ls_size, best_coll = _evaluate(best_arr)
    
    random.seed(12345)
    
    # Diverse initialization strategies
    def make_ap_start():
        """Arithmetic progression based start - many collisions."""
        while True:
            a = random.randint(0, 20)
            d = random.randint(2, 15)
            vals = [a + i * d for i in range(12)]
            if max(vals) <= 128 and min(vals) >= 0:
                return sorted(set(vals)) if len(set(vals)) == 12 else None
    
    def make_cluster_start():
        """Clustered values - forces collisions."""
        center = random.randint(20, 110)
        vals = []
        for _ in range(12):
            v = center + random.randint(-15, 15)
            v = max(0, min(128, v))
            vals.append(v)
        # Ensure distinct
        if len(set(vals)) < 12:
            return None
        return sorted(vals)
    
    def make_random_start():
        return sorted(random.sample(range(129), 12))
    
    def make_mixed_start():
        """Mix of AP and random."""
        a = random.randint(0, 15)
        d = random.randint(3, 12)
        vals = []
        for i in range(12):
            if random.random() < 0.5:
                v = a + i * d
            else:
                v = random.randint(0, 128)
            v = max(0, min(128, v))
            vals.append(v)
        if len(set(vals)) < 12:
            return None
        return sorted(vals)
    
    starts = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    ]
    
    for _ in range(30):
        s = make_ap_start()
        if s:
            starts.append(s)
    for _ in range(30):
        s = make_cluster_start()
        if s:
            starts.append(s)
    for _ in range(30):
        s = make_random_start()
        if s:
            starts.append(s)
    for _ in range(30):
        s = make_mixed_start()
        if s:
            starts.append(s)
    
    # Greedy local search
    def greedy_search(arr, max_time=None):
        """Greedy local search to minimize largest Sidon subset size."""
        arr = sorted(arr)
        current_ls, current_coll = _evaluate(arr)
        
        improved = True
        while improved:
            if max_time and time.time() > max_time:
                break
            improved = False
            for i in range(12):
                if max_time and time.time() > max_time:
                    break
                for new_val in range(129):
                    if new_val in arr:
                        continue
                    new_arr = arr[:i] + [new_val] + arr[i+1:]
                    new_arr = sorted(new_arr)
                    new_ls, new_coll = _evaluate(new_arr)
                    # Prefer smaller largest Sidon size; break ties with more collisions
                    if new_ls < current_ls or (new_ls == current_ls and new_coll > current_coll):
                        arr = new_arr
                        current_ls, current_coll = new_ls, new_coll
                        improved = True
                        break
        return arr, current_ls, current_coll
    
    # Try each start with time budget
    num_starts = len(starts)
    time_per_start = 70.0 / num_starts  # ~70s total budget
    
    for idx, start_arr in enumerate(starts):
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        
        # Allocate time for this start
        allocated = min(time_per_start, time_left - 1.0)
        if allocated < 0.5:
            break
        
        end_time = time.time() + allocated
        result_arr, ls_size, coll = greedy_search(start_arr, max_time=end_time)
        
        if ls_size < best_ls_size or (ls_size == best_ls_size and coll > best_coll):
            best_arr = result_arr
            best_ls_size = ls_size
            best_coll = coll
    
    # Final polish: try a few more targeted improvements on the best
    if time.time() < deadline - 5:
        result_arr, ls_size, coll = greedy_search(best_arr, max_time=deadline - 2)
        if ls_size < best_ls_size or (ls_size == best_ls_size and coll > best_coll):
            best_arr = result_arr
            best_ls_size = ls_size
            best_coll = coll
    
    return best_arr

# EVOLVE-BLOCK-END
