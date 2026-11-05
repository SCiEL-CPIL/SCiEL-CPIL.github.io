"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def _collision_count(arr):
    """Count duplicate pair sums using NumPy vectorization. Higher = more collisions."""
    a = np.array(arr, dtype=np.int32)
    n = len(a)
    # Compute all pair sums i <= j
    i_idx, j_idx = np.triu_indices(n)
    sums = a[i_idx] + a[j_idx]
    # Count unique sums
    unique_count = len(np.unique(sums))
    total_pairs = n * (n + 1) // 2
    return total_pairs - unique_count

def _is_sidon(subset):
    """Check if a subset has all unique pair sums (including diagonals)."""
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
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if _is_sidon(subset):
                return k
    return 0

def _evaluate_exact(arr):
    """Exact evaluation: return max Sidon subset size."""
    return _max_sidon_subset_size(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 55

    best_arr = list(range(12))
    best_exact_size = _evaluate_exact(best_arr)
    best_collisions = _collision_count(best_arr)

    random.seed(42)
    np.random.seed(42)

    # Phase 1: Fast surrogate-based search using collision count
    # We want to MAXIMIZE collisions to minimize Sidon subset size
    
    all_candidates = []
    
    for restart in range(5000):
        if time.time() > deadline:
            break

        # Multiple initialization strategies
        strategy = restart % 5
        if strategy == 0:
            arr = sorted(random.sample(range(129), 12))
        elif strategy == 1:
            # 4 clusters of 3 - tight clusters create many collisions
            base = random.randint(0, 90)
            gap = random.randint(8, 12)
            arr = []
            for i in range(4):
                offset = base + i * gap
                arr.extend([offset, offset+1, offset+2])
            if max(arr) > 128:
                continue
            arr = sorted(arr)
        elif strategy == 2:
            # 6 pairs with small offsets
            base = random.randint(0, 100)
            gap = random.randint(10, 14)
            arr = []
            for i in range(6):
                offset = base + i * gap
                arr.extend([offset, offset + random.randint(1, 3)])
            if len(set(arr)) < 12 or max(arr) > 128:
                continue
            arr = sorted(arr)
        elif strategy == 3:
            # 3 groups of 4 - very tight
            base = random.randint(0, 80)
            gap = random.randint(12, 16)
            arr = []
            for i in range(3):
                offset = base + i * gap
                arr.extend([offset, offset+1, offset+2, offset+3])
            if max(arr) > 128:
                continue
            arr = sorted(arr)
        else:
            # Random with bias toward clustering
            arr = sorted(random.sample(range(129), 12))
            # Perturb to create more collisions
            for _ in range(3):
                i = random.randint(0, 11)
                j = random.randint(0, 11)
                if i != j:
                    diff = arr[j] - arr[i]
                    if abs(diff) <= 10 and diff != 0:
                        new_val = arr[i] + diff
                        if 0 <= new_val <= 128 and new_val not in arr:
                            arr[i] = new_val
                            arr = sorted(arr)

        if len(set(arr)) != 12 or min(arr) < 0 or max(arr) > 128:
            continue

        current_collisions = _collision_count(arr)

        # Local search using fast collision count
        no_improve = 0
        for step in range(500):
            if time.time() > deadline:
                break

            move_type = random.random()
            if move_type < 0.4:
                # Single replacement
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            elif move_type < 0.65:
                # Double replacement
                i, j = random.sample(range(12), 2)
                nv1 = random.randint(0, 128)
                nv2 = random.randint(0, 128)
                if nv1 in arr or nv2 in arr or nv1 == nv2:
                    continue
                new_arr = arr.copy()
                new_arr[i] = nv1
                new_arr[j] = nv2
            elif move_type < 0.85:
                # Small perturbation - push elements closer together
                i = random.randint(0, 11)
                # Try to move toward another element to create collisions
                j = random.randint(0, 11)
                if i == j:
                    continue
                delta = random.choice([-3,-2,-1,1,2,3])
                new_val = arr[i] + delta
                if new_val < 0 or new_val > 128 or new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            else:
                # Swap two elements with nearby values
                i, j = random.sample(range(12), 2)
                nv1 = arr[i] + random.randint(-5, 5)
                nv2 = arr[j] + random.randint(-5, 5)
                if nv1 < 0 or nv1 > 128 or nv2 < 0 or nv2 > 128:
                    continue
                if nv1 in arr or nv2 in arr or nv1 == nv2:
                    continue
                new_arr = arr.copy()
                new_arr[i] = nv1
                new_arr[j] = nv2

            new_arr = sorted(new_arr)
            if len(set(new_arr)) != 12:
                continue

            new_collisions = _collision_count(new_arr)

            if new_collisions > current_collisions:
                arr = new_arr
                current_collisions = new_collisions
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 100:
                    break

        all_candidates.append((current_collisions, arr))

    # Phase 2: Evaluate top candidates exactly
    # Sort by collision count (descending) and evaluate exact Sidon size
    all_candidates.sort(key=lambda x: -x[0])
    
    evaluated = set()
    for collisions, arr in all_candidates[:50]:
        if time.time() > deadline:
            break
        # Check if we've seen this set before
        arr_tuple = tuple(arr)
        if arr_tuple in evaluated:
            continue
        evaluated.add(arr_tuple)
        
        exact_size = _evaluate_exact(arr)
        if exact_size < best_exact_size:
            best_exact_size = exact_size
            best_arr = arr
            if best_exact_size <= 4:
                return best_arr

    # Phase 3: Final polish - local search on best exact solution
    if best_exact_size > 4:
        arr = best_arr.copy()
        current_exact = best_exact_size
        no_improve = 0
        for step in range(2000):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in arr:
                continue
            new_arr = arr.copy()
            new_arr[i] = new_val
            new_arr = sorted(new_arr)
            
            if len(set(new_arr)) != 12:
                continue
            
            new_exact = _evaluate_exact(new_arr)
            if new_exact < current_exact:
                arr = new_arr
                current_exact = new_exact
                no_improve = 0
                if current_exact < best_exact_size:
                    best_exact_size = current_exact
                    best_arr = arr
                    if best_exact_size <= 4:
                        return best_arr
            else:
                no_improve += 1
                if no_improve > 200:
                    break

    return best_arr
# EVOLVE-BLOCK-END
