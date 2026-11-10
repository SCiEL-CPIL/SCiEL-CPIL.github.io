import time
import random
import numpy as np

N = 512
DEADLINE = 85.0  # safety margin before 90s kill

def _has_4ap(arr, selected_mask):
    """Check if adding arr (scalar or array) to selected_mask creates a 4-AP.
    selected_mask is a boolean numpy array of length N+1 (index 0 unused).
    Returns True if any nonconstant 4-AP exists involving arr elements.
    """
    # For each candidate x in arr, check all d such that x-3d >= 1
    # and verify x-d, x-2d, x-3d are all in selected_mask
    # Also check if x is part of a 4-AP as the first, second, or third term
    # where the 4th term is already selected.
    
    # Efficient approach: for each x in arr, for each d, check patterns
    # Pattern 1: x is 4th term: x-3d, x-2d, x-d all selected
    # Pattern 2: x is 3rd term: x-2d, x-d, x+d all selected (x+d <= N)
    # Pattern 3: x is 2nd term: x-d, x+d, x+2d all selected (x+2d <= N)
    # Pattern 4: x is 1st term: x+d, x+2d, x+3d all selected (x+3d <= N)
    
    for x in arr:
        x = int(x)
        # Pattern 1: x is 4th term
        max_d1 = (x - 1) // 3
        for d in range(1, max_d1 + 1):
            if selected_mask[x - d] and selected_mask[x - 2*d] and selected_mask[x - 3*d]:
                return True
        # Pattern 2: x is 3rd term
        max_d2 = min((x - 1) // 2, N - x)
        for d in range(1, max_d2 + 1):
            if selected_mask[x - 2*d] and selected_mask[x - d] and selected_mask[x + d]:
                return True
        # Pattern 3: x is 2nd term
        max_d3 = min(x - 1, (N - x) // 2)
        for d in range(1, max_d3 + 1):
            if selected_mask[x - d] and selected_mask[x + d] and selected_mask[x + 2*d]:
                return True
        # Pattern 4: x is 1st term
        max_d4 = (N - x) // 3
        for d in range(1, max_d4 + 1):
            if selected_mask[x + d] and selected_mask[x + 2*d] and selected_mask[x + 3*d]:
                return True
    return False


def _greedy_build(order, start_time, deadline):
    """Build a 4-AP-free set by processing elements in 'order' and adding if no conflict."""
    selected = np.zeros(N + 1, dtype=bool)
    for x in order:
        if time.time() > deadline:
            break
        x = int(x)
        if selected[x]:
            continue
        if not _has_4ap(np.array([x]), selected):
            selected[x] = True
    return selected


def _local_search(selected, start_time, deadline, max_rounds=50):
    """Try to improve via swap moves: remove one element, add one or more others."""
    best_size = int(np.sum(selected))
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        if time.time() > deadline:
            break
        # Try single swap: remove i, add j
        current_indices = np.where(selected)[0]
        # Shuffle for randomization
        rng = random.Random(rounds * 7919 + 13)
        indices_list = list(current_indices)
        rng.shuffle(indices_list)
        
        for i in indices_list:
            if time.time() > deadline:
                return selected, best_size
            # Try removing i
            selected[i] = False
            # Try adding each non-selected j
            non_selected = np.where(~selected)[0]
            rng.shuffle(non_selected.tolist())
            added_any = False
            for j in non_selected:
                j = int(j)
                if selected[j]:
                    continue
                if not _has_4ap(np.array([j]), selected):
                    selected[j] = True
                    added_any = True
                    break
            if added_any:
                # We removed i and added j, size unchanged. Try to also add more.
                # Actually this is a swap that keeps size same. Let's try refill.
                # Try to add more elements now that we have a "hole"
                non_selected2 = np.where(~selected)[0]
                rng.shuffle(non_selected2.tolist())
                for j in non_selected2:
                    j = int(j)
                    if not _has_4ap(np.array([j]), selected):
                        selected[j] = True
                new_size = int(np.sum(selected))
                if new_size > best_size:
                    best_size = new_size
                    improved = True
                    break
                else:
                    # Revert: remove j's we added, restore i
                    # This is getting complex. Let's simplify.
                    pass
            else:
                # Couldn't add anything, restore i
                selected[i] = True
        # After trying all removals, try pure additions
        non_selected = np.where(~selected)[0]
        rng.shuffle(non_selected.tolist())
        for j in non_selected:
            j = int(j)
            if not _has_4ap(np.array([j]), selected):
                selected[j] = True
                best_size = int(np.sum(selected))
                improved = True
    return selected, best_size


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_set = None
    best_size = 0
    
    # Multiple random greedy starts
    num_starts = 20
    for s in range(num_starts):
        if time.time() > deadline - 10:
            break
        # Random ordering
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Also try some heuristic orderings
        if s % 5 == 0:
            # Descending
            order = list(range(N, 0, -1))
        elif s % 5 == 1:
            # Ascending
            order = list(range(1, N + 1))
        elif s % 5 == 2:
            # Random
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif s % 5 == 3:
            # Primes first
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in range(1, N + 1) if is_prime(x)]
            non_primes = [x for x in range(1, N + 1) if not is_prime(x)]
            random.shuffle(non_primes)
            order = primes + non_primes
        elif s % 5 == 4:
            # Multiples of 2 first, then others
            evens = [x for x in range(1, N + 1) if x % 2 == 0]
            odds = [x for x in range(1, N + 1) if x % 2 != 0]
            random.shuffle(evens)
            random.shuffle(odds)
            order = evens + odds
        
        selected = _greedy_build(order, start_time, deadline)
        size = int(np.sum(selected))
        
        if size > best_size:
            best_size = size
            best_set = selected.copy()
        
        # Local search refinement
        if time.time() < deadline - 15:
            selected, new_size = _local_search(selected.copy(), start_time, deadline, max_rounds=20)
            if new_size > best_size:
                best_size = new_size
                best_set = selected.copy()
    
    # Final attempt: try to add more to best_set
    if best_set is not None and time.time() < deadline - 5:
        non_selected = np.where(~best_set)[0]
        random.shuffle(non_selected.tolist())
        for j in non_selected:
            j = int(j)
            if not _has_4ap(np.array([j]), best_set):
                best_set[j] = True
        best_size = int(np.sum(best_set))
    
    if best_set is None or best_size == 0:
        # Fallback: return at least one element
        return [1]
    
    result = sorted(int(x) for x in np.where(best_set)[0])
    return result if result else [1]
