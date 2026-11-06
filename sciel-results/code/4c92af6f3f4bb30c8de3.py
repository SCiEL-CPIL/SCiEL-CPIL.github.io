"""B_3[1] construction: batched bitset-based collision checking for parallel candidate evaluation."""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 82.0
    N = 4096
    MAX_SUM = 3 * N  # 12288

    def build_sums(arr):
        """Build boolean array of all triple sums (with repetition, unordered)."""
        n = len(arr)
        sums = np.zeros(MAX_SUM + 1, dtype=np.bool_)
        if n == 0:
            return sums
        # Use broadcasting for all triples i <= j <= k
        # For efficiency, do it in chunks if n is large
        for i in range(n):
            a = arr[i]
            for j in range(i, n):
                b = arr[j]
                for k in range(j, n):
                    c = arr[k]
                    sums[a + b + c] = True
        return sums

    def can_add_batched(x, sums_arr, selected_np):
        """Check if x can be added to the set with given sums."""
        # 3x
        if sums_arr[3 * x]:
            return False
        n = len(selected_np)
        if n == 0:
            return True
        # 2x + s
        if np.any(sums_arr[2 * x + selected_np]):
            return False
        # x + 2s
        if np.any(sums_arr[x + 2 * selected_np]):
            return False
        # x + s + t for s <= t
        i_idx, j_idx = np.triu_indices(n)
        if np.any(sums_arr[x + selected_np[i_idx] + selected_np[j_idx]]):
            return False
        return True

    def add_to_sums(x, sums_arr, selected_np):
        """Update sums array to include x in the set."""
        sums_arr[3 * x] = True
        n = len(selected_np)
        if n == 0:
            return
        sums_arr[2 * x + selected_np] = True
        sums_arr[x + 2 * selected_np] = True
        i_idx, j_idx = np.triu_indices(n)
        sums_arr[x + selected_np[i_idx] + selected_np[j_idx]] = True

    # Phase 1: Greedy from low to high with vectorized checks
    selected = []
    selected_np = np.array([], dtype=np.int32)
    sums_arr = np.zeros(MAX_SUM + 1, dtype=np.bool_)
    sel_set = set()

    for x in range(1, N + 1):
        if time.time() > deadline - 25.0:
            break
        if can_add_batched(x, sums_arr, selected_np):
            selected.append(x)
            selected_np = np.append(selected_np, x)
            sel_set.add(x)
            add_to_sums(x, sums_arr, selected_np[:-1])

    best_selected = selected[:]
    best_size = len(selected)

    # Phase 2: Batched swap improvement
    # Try to replace elements with larger ones to free up sum space
    if time.time() < deadline - 15.0:
        sel_set = set(selected)
        candidates = [x for x in range(N, 0, -1) if x not in sel_set]
        candidates.sort(reverse=True)

        improved = True
        while improved and time.time() < deadline - 12.0:
            improved = False
            for i in range(len(selected) - 1, -1, -1):
                if time.time() > deadline - 12.0:
                    break
                orig = selected[i]
                # Try to find a larger candidate that can replace orig
                for cand in candidates[:50]:  # Limit candidates to try per position
                    if cand <= orig:
                        break
                    # Temporarily remove orig
                    temp_selected = selected[:i] + selected[i+1:]
                    temp_np = np.array(temp_selected, dtype=np.int32)
                    temp_sums = build_sums(temp_np)
                    
                    if can_add_batched(cand, temp_sums, temp_np):
                        selected[i] = cand
                        selected_np = np.array(selected, dtype=np.int32)
                        selected.sort()
                        candidates.remove(cand)
                        improved = True
                        break
                if time.time() > deadline - 12.0:
                    break

        if len(selected) > best_size:
            best_selected = selected[:]
            best_size = len(selected)

    # Phase 3: Randomized greedy restarts with different orderings
    if time.time() < deadline - 5.0:
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Create a randomized ordering of candidates
            sel_set = set(best_selected)
            all_candidates = [x for x in range(1, N + 1) if x not in sel_set]
            random.shuffle(all_candidates)
            
            # Start from best and try to add more
            current = best_selected[:]
            current_np = np.array(current, dtype=np.int32)
            current_sums = build_sums(current_np)
            
            for x in all_candidates:
                if time.time() > deadline - 5.0:
                    break
                if can_add_batched(x, current_sums, current_np):
                    current.append(x)
                    current_np = np.append(current_np, x)
                    add_to_sums(x, current_sums, current_np[:-1])
            
            if len(current) > best_size:
                best_selected = current[:]
                best_size = len(current)

    return best_selected[:80]
