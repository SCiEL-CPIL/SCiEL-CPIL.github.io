import time
import numpy as np

N = 256
DEADLINE = 75.0  # safety margin under 90s wall-clock

def _count_3aps(arr):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0 in boolean array arr over [1..N]."""
    n = len(arr)
    if n < 3:
        return 0
    # For each d, arr[i] & arr[i+d] & arr[i+2d] summed over i
    total = 0
    for d in range(1, (n - 2) // 2 + 1):
        total += int(np.sum(arr[:n - 2 * d] & arr[d:n - d] & arr[2 * d:n]))
    return total

def _has_4ap(arr):
    """Return True if arr (bool over [1..N]) contains a nonconstant 4-term AP."""
    n = len(arr)
    # 4-AP: a, a+d, a+2d, a+3d with d>=1
    for d in range(1, (n - 3) // 3 + 1):
        if np.any(arr[:n - 3 * d] & arr[d:n - 2 * d] & arr[2 * d:n - d] & arr[3 * d:n]):
            return True
    return False

def _valid_add(arr, pos):
    """Check if adding pos (1-indexed) to arr keeps it 4-free. pos not in arr."""
    idx = pos - 1
    if arr[idx]:
        return False
    # A new 4-AP must include pos. pos can be at any of 4 positions in the 4-AP.
    # Case 1: pos = a (first term): need pos+d, pos+2d, pos+3d in arr
    for d in range(1, (N - pos) // 3 + 1):
        if arr[pos - 1 + d] and arr[pos - 1 + 2 * d] and arr[pos - 1 + 3 * d]:
            return False
    # Case 2: pos = a+d (second term): need pos-d, pos+d, pos+2d in arr
    for d in range(1, min(pos - 1, (N - pos) // 2 + 1) + 1):
        if arr[pos - 1 - d] and arr[pos - 1 + d] and arr[pos - 1 + 2 * d]:
            return False
    # Case 3: pos = a+2d (third term): need pos-2d, pos-d, pos+d in arr
    for d in range(1, min((pos - 1) // 2, N - pos) + 1):
        if arr[pos - 1 - 2 * d] and arr[pos - 1 - d] and arr[pos - 1 + d]:
            return False
    # Case 4: pos = a+3d (fourth term): need pos-3d, pos-2d, pos-d in arr
    for d in range(1, (pos - 1) // 3 + 1):
        if arr[pos - 1 - 3 * d] and arr[pos - 1 - 2 * d] and arr[pos - 1 - d]:
            return False
    return True

def _valid_remove(arr, pos):
    """Removing always keeps 4-free (removing can't create a 4-AP). Always valid."""
    return True

def _valid_swap(arr, pos_in, pos_out):
    """Check if swapping pos_out (remove) and pos_in (add) keeps 4-free."""
    idx_in = pos_in - 1
    idx_out = pos_out - 1
    if arr[idx_in] or not arr[idx_out]:
        return False
    # Temporarily remove pos_out and add pos_in
    arr[idx_out] = False
    arr[idx_in] = True
    valid = not _has_4ap(arr)
    arr[idx_in] = False
    arr[idx_out] = True
    return valid

def _greedy_construct():
    """Greedy: add each x from 1..N if it doesn't create a 4-AP."""
    arr = np.zeros(N, dtype=bool)
    for x in range(1, N + 1):
        if _valid_add(arr, x):
            arr[x - 1] = True
    return arr

def _local_search(arr, time_limit):
    """Local search: single-add, single-remove, swap moves to maximize 3-AP count."""
    best_arr = arr.copy()
    best_count = _count_3aps(best_arr)
    deadline = time.time() + time_limit
    no_improve_iters = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_iters < max_no_improve:
        improved = False
        current_count = best_count
        
        # Try single additions
        for x in range(1, N + 1):
            if not best_arr[x - 1] and _valid_add(best_arr, x):
                best_arr[x - 1] = True
                new_count = _count_3aps(best_arr)
                if new_count > current_count:
                    current_count = new_count
                    improved = True
                else:
                    best_arr[x - 1] = False
        
        # Try single removals (only if it helps 3-AP count, which is unusual but possible)
        for x in range(1, N + 1):
            if best_arr[x - 1]:
                best_arr[x - 1] = False
                new_count = _count_3aps(best_arr)
                if new_count > current_count:
                    current_count = new_count
                    improved = True
                else:
                    best_arr[x - 1] = True
        
        # Try swaps: remove one, add another
        removed = []
        added = []
        for x_out in range(1, N + 1):
            if not best_arr[x_out - 1]:
                continue
            for x_in in range(1, N + 1):
                if best_arr[x_in - 1]:
                    continue
                if _valid_swap(best_arr, x_in, x_out):
                    best_arr[x_out - 1] = False
                    best_arr[x_in - 1] = True
                    new_count = _count_3aps(best_arr)
                    if new_count > current_count:
                        current_count = new_count
                        improved = True
                        removed.append(x_out)
                        added.append(x_in)
                    else:
                        best_arr[x_in - 1] = False
                        best_arr[x_out - 1] = True
        
        if improved:
            best_count = current_count
            no_improve_iters = 0
        else:
            no_improve_iters += 1
            
            # If stuck, try a random restart perturbation
            if no_improve_iters % 5 == 0:
                # Remove 2-3 random elements and re-greedy fill
                idxs = np.where(best_arr)[0]
                if len(idxs) > 10:
                    k = min(3, len(idxs) // 10)
                    remove_idxs = np.random.choice(idxs, size=k, replace=False)
                    for ri in remove_idxs:
                        best_arr[ri] = False
                    # Re-greedy: try to add back any missing elements
                    for x in range(1, N + 1):
                        if not best_arr[x - 1] and _valid_add(best_arr, x):
                            best_arr[x - 1] = True
                    new_count = _count_3aps(best_arr)
                    if new_count >= best_count:
                        best_count = new_count
                        no_improve_iters = 0
                        improved = True
        
        if time.time() >= deadline:
            break
    
    return best_arr, best_count

def construct():
    t_start = time.time()
    
    # Phase 1: Greedy construction
    arr = _greedy_construct()
    count = _count_3aps(arr)
    
    # Phase 2: Local search with time budget
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 2.0:
        arr, count = _local_search(arr, remaining - 1.0)
    
    # Safety: ensure no 4-AP
    if _has_4ap(arr):
        # Fallback: rebuild greedily
        arr = _greedy_construct()
    
    result = sorted(int(i + 1) for i in np.where(arr)[0])
    if not result:
        result = [1]
    
    return result
