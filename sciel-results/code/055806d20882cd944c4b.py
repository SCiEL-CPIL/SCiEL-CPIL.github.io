"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
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

def _count_duplicate_sums(arr):
    """Count total number of duplicate pair sum occurrences (higher = more collisions)."""
    sums = []
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            sums.append(arr[i] + arr[j])
    c = Counter(sums)
    dup_count = 0
    for count in c.values():
        if count > 1:
            dup_count += count - 1
    return dup_count

def _evaluate(arr):
    """Evaluate: return (max_sidon_size, -duplicate_count) for minimization."""
    size = _max_sidon_subset_size(arr)
    dup = _count_duplicate_sums(arr)
    return (size, -dup)

def construct():
    start_time = time.time()
    deadline = start_time + 55

    best_arr = list(range(12))
    best_eval = _evaluate(best_arr)

    random.seed(42)

    for restart in range(800):
        if time.time() > deadline:
            break

        # Multiple initialization strategies
        strategy = restart % 4
        if strategy == 0:
            arr = sorted(random.sample(range(129), 12))
        elif strategy == 1:
            # 4 clusters of 3
            base = random.randint(0, 85)
            gap = random.randint(10, 14)
            arr = []
            for i in range(4):
                offset = base + i * gap
                arr.extend([offset, offset+1, offset+2])
            if max(arr) > 128:
                continue
            arr = sorted(arr)
        elif strategy == 2:
            # 6 pairs
            base = random.randint(0, 95)
            gap = random.randint(12, 18)
            arr = []
            for i in range(6):
                offset = base + i * gap
                arr.extend([offset, offset + random.randint(1, 4)])
            if len(set(arr)) < 12 or max(arr) > 128:
                continue
            arr = sorted(arr)
        else:
            # 3 groups of 4
            base = random.randint(0, 75)
            gap = random.randint(14, 20)
            arr = []
            for i in range(3):
                offset = base + i * gap
                arr.extend([offset, offset+1, offset+2, offset+3])
            if max(arr) > 128:
                continue
            arr = sorted(arr)

        current_eval = _evaluate(arr)

        if current_eval < best_eval:
            best_arr = arr
            best_eval = current_eval
            if best_eval[0] <= 4:
                return best_arr

        # Local search
        no_improve = 0
        for step in range(3000):
            if time.time() > deadline:
                break

            move_type = random.random()
            if move_type < 0.35:
                # Single replacement
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            elif move_type < 0.6:
                # Double replacement
                i, j = random.sample(range(12), 2)
                nv1 = random.randint(0, 128)
                nv2 = random.randint(0, 128)
                if nv1 in arr or nv2 in arr or nv1 == nv2:
                    continue
                new_arr = arr.copy()
                new_arr[i] = nv1
                new_arr[j] = nv2
            elif move_type < 0.8:
                # Small perturbation
                i = random.randint(0, 11)
                delta = random.choice([-4,-3,-2,-1,1,2,3,4,6,8])
                new_val = arr[i] + delta
                if new_val < 0 or new_val > 128 or new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            else:
                # Swap two elements with new values
                i, j = random.sample(range(12), 2)
                nv1 = random.randint(0, 128)
                nv2 = random.randint(0, 128)
                if nv1 in arr or nv2 in arr or nv1 == nv2:
                    continue
                new_arr = arr.copy()
                new_arr[i] = nv1
                new_arr[j] = nv2

            new_arr = sorted(new_arr)
            if len(set(new_arr)) != 12:
                continue

            new_eval = _evaluate(new_arr)

            if new_eval < current_eval:
                arr = new_arr
                current_eval = new_eval
                no_improve = 0
                if current_eval < best_eval:
                    best_arr = arr
                    best_eval = current_eval
                    if best_eval[0] <= 4:
                        return best_arr
            else:
                no_improve += 1
                if no_improve > 150:
                    break

    return best_arr
# EVOLVE-BLOCK-END
