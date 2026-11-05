"""Sidon set extension with greedy + swap local search + polish."""
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 140s safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    
    def compute_sums(arr):
        s = set()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                s.add(arr[i] + arr[j])
        return s

    def can_add(arr, sums, x):
        """Check if x can be added to arr maintaining Sidon property."""
        # x must not be in arr
        if x in arr:
            return False
        # Check all pairs (x, y) for y in arr, and (x, x)
        for y in arr:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def try_add(arr, sums, x):
        """Add x to arr and update sums. Returns True if successful."""
        if not can_add(arr, sums, x):
            return False
        new_sums = set()
        for y in arr:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        arr.append(x)
        sums.update(new_sums)
        return True

    def greedy_construct():
        """Greedy construction starting from prescribed elements."""
        arr = list(prescribed)
        sums = compute_sums(arr)
        # Sort candidates to prefer smaller values first (more room)
        for x in range(12, 285):
            if try_add(arr, sums, x):
                pass
        return arr, sums

    def swap_local_search(arr, sums, max_time=None):
        """Try single swaps: remove one non-prescribed element, add a better one."""
        if max_time is None:
            max_time = time.time() + 5
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try removing each non-prescribed element
            for idx in range(len(arr) - 1, -1, -1):
                if arr[idx] in prescribed:
                    continue
                removed = arr[idx]
                # Try adding each possible replacement
                for x in range(1, 285):
                    if x in arr or x == removed:
                        continue
                    # Temporarily remove
                    new_arr = arr[:idx] + arr[idx+1:]
                    new_sums = compute_sums(new_arr)
                    if try_add(new_arr, new_sums, x):
                        # Accept the improvement (size stays same, but may enable further growth)
                        arr = new_arr
                        sums = new_sums
                        improved = True
                        break
                if improved:
                    break
        # After swaps, try to grow again
        for x in range(1, 285):
            if try_add(arr, sums, x):
                pass
        return arr, sums

    def polish(arr, sums):
        """Exhaustive single-swap polish to find a locally optimal set."""
        improved = True
        while improved:
            improved = False
            for idx in range(len(arr)):
                if arr[idx] in prescribed:
                    continue
                removed = arr[idx]
                for x in range(1, 285):
                    if x in arr or x == removed:
                        continue
                    new_arr = arr[:idx] + arr[idx+1:]
                    new_sums = compute_sums(new_arr)
                    if try_add(new_arr, new_sums, x):
                        arr = new_arr
                        sums = new_sums
                        improved = True
                        break
                if improved:
                    break
        return arr, sums

    # Phase 1: Greedy construction
    arr, sums = greedy_construct()
    best_size = len(arr)
    best_arr = list(arr)
    best_sums = set(sums)

    # Phase 2: Swap local search to improve
    if time.time() < DEADLINE:
        arr, sums = swap_local_search(arr, sums, max_time=min(DEADLINE - 5, time.time() + 20))
        if len(arr) > best_size:
            best_size = len(arr)
            best_arr = list(arr)
            best_sums = set(sums)

    # Phase 3: Polish the best solution
    if time.time() < DEADLINE:
        arr, sums = polish(best_arr, best_sums)
        if len(arr) > best_size:
            best_size = len(arr)
            best_arr = list(arr)
            best_sums = set(sums)

    # Phase 4: Try multiple greedy starts with different orderings
    if time.time() < DEADLINE:
        import random
        random.seed(42)
        for trial in range(50):
            if time.time() >= DEADLINE - 10:
                break
            # Randomized greedy: shuffle candidate order
            arr = list(prescribed)
            sums = compute_sums(arr)
            candidates = list(range(12, 285))
            random.shuffle(candidates)
            for x in candidates:
                if try_add(arr, sums, x):
                    pass
            if len(arr) > best_size:
                # Improve with swap search
                arr, sums = swap_local_search(arr, sums, max_time=min(DEADLINE - 5, time.time() + 5))
                if len(arr) > best_size:
                    best_size = len(arr)
                    best_arr = list(arr)
                    best_sums = set(sums)
                # Polish
                arr, sums = polish(best_arr, best_sums)
                if len(arr) > best_size:
                    best_size = len(arr)
                    best_arr = list(arr)
                    best_sums = set(sums)

    return best_arr
