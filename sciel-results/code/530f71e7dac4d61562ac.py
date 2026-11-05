import time
import random
from itertools import combinations

def _is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr):
    n = len(arr)
    arr_sorted = sorted(arr)
    # Check sizes from n down to 4
    for size in range(n, 3, -1):
        # For size 4, we assume it's the minimum possible we care about
        # If we find a Sidon subset of size 5, return 5
        # If we check all subsets of size 5 and none is Sidon, return 4
        if size == 4:
            return 4
        # Check if any subset of this size is Sidon
        for combo in combinations(arr_sorted, size):
            if _is_sidon(combo):
                return size
    return 4

def _count_additive_quadruples(arr):
    n = len(arr)
    count = 0
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                count += sums[s]
            else:
                sums[s] = 1
    return count

def construct():
    start_time = time.time()
    deadline = 80
    
    # Initial candidates
    candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
    ]
    
    best_arr = None
    best_energy = -1
    best_sidon_size = 12
    
    for arr in candidates:
        if time.time() - start_time > deadline:
            break
        energy = _count_additive_quadruples(arr)
        sidon_size = _max_sidon_subset_size(arr)
        if sidon_size < best_sidon_size or (sidon_size == best_sidon_size and energy > best_energy):
            best_sidon_size = sidon_size
            best_energy = energy
            best_arr = list(arr)
        if best_sidon_size == 4:
            return best_arr

    # Local search
    current = list(best_arr)
    current_energy = best_energy
    
    while time.time() - start_time < deadline:
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
                new_energy = _count_additive_quadruples(new_arr)
                if new_energy > current_energy:
                    current = new_arr
                    current_energy = new_energy
                    # Check sidon size occasionally or if energy jump is big
                    if current_energy > best_energy:
                        sidon_size = _max_sidon_subset_size(current)
                        if sidon_size < best_sidon_size:
                            best_sidon_size = sidon_size
                            best_arr = list(current)
                            if best_sidon_size == 4:
                                return best_arr
                        elif sidon_size == best_sidon_size:
                            best_energy = current_energy
                            best_arr = list(current)
                    improved = True
                    break
                elif new_energy == current_energy and random.random() < 0.05:
                    current = new_arr
                    improved = True
                    break
        if not improved:
            if time.time() - start_time > deadline - 5:
                break
            # Perturb
            for _ in range(3):
                idx = random.randint(0, 11)
                current[idx] = random.randint(0, 128)
            if len(set(current)) != 12:
                continue
            current_energy = _count_additive_quadruples(current)
            
    return best_arr
