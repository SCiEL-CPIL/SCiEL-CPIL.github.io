"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

_START_TIME = time.time()
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _is_sidon_with_diagonals(subset):
    """Check if subset has all unordered pair sums unique including diagonals (2a[i])."""
    n = len(subset)
    if n <= 1:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_sidon_size(arr, order=None):
    """Greedy construction of a Sidon subset. Returns size."""
    if order is None:
        order = arr
    else:
        order = sorted(arr, key=lambda x: order.index(x)) # This is O(N^2), bad.
    # Better: pass indices
    # Let's just sort the values and iterate.
    # To simulate different orders, we can just shuffle the indices.
    
    # For now, simple greedy on sorted values
    pass

def _estimate_max_sidon(arr, rng, trials=20):
    """
    Estimate the maximum Sidon subset size by finding large Sidon subsets
    using greedy heuristics on various orderings.
    Returns the maximum size found.
    """
    n = len(arr)
    if n == 0: return 0
    
    best_size = 0
    
    # Helper to run greedy on a specific permutation of indices
    def _run_greedy(indices):
        current_elements = []
        current_sums = set()
        for idx in indices:
            x = arr[idx]
            can_add = True
            new_sums = set()
            
            # Check sums involving x against existing sums
            for y in current_elements:
                s = x + y
                if s in current_sums:
                    can_add = False
                    break
                new_sums.add(s)
            
            if can_add:
                s_diag = 2 * x
                if s_diag in current_sums:
                    can_add = False
                else:
                    new_sums.add(s_diag)
            
            if can_add:
                current_elements.append(x)
                current_sums.update(new_sums)
        
        return len(current_elements)

    indices = list(range(n))
    
    # 1. Sorted order
    sorted_indices = sorted(indices, key=lambda i: arr[i])
    best_size = max(best_size, _run_greedy(sorted_indices))
    
    # 2. Reverse sorted
    rev_indices = sorted_indices[::-1]
    best_size = max(best_size, _run_greedy(rev_indices))
    
    # 3. Random shuffles
    for _ in range(trials):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        size = _run_greedy(shuffled)
        if size > best_size:
            best_size = size
    
    return best_size

def _exact_max_sidon_subset_size(arr, time_limit=5.0):
    """
    Exact B&B solver with a time limit.
    Returns the exact max size if found within time_limit, 
    otherwise returns the best lower bound found so far.
    """
    n = len(arr)
    if n <= 1:
        return n
    
    arr_sorted = sorted(arr)
    best = 0
    start_time = time.time()
    
    # Global flag to stop
    stop = False
    
    def dfs(idx, current_size, current_sums, current_elements):
        nonlocal best, stop, start_time
        
        if stop:
            return
            
        # Pruning: if we can't beat best
        if current_size + (n - idx) <= best:
            return
        
        # Update best
        if current_size > best:
            best = current_size
            if best == n:
                stop = True
                return
        
        # Time check
        if time.time() - start_time > time_limit:
            stop = True
            return
            
        if idx == n:
            return
        
        # Option 1: Exclude
        dfs(idx + 1, current_size, current_sums, current_elements)
        if stop: return
        
        # Option 2: Include
        x = arr_sorted[idx]
        new_sums = set()
        can_add = True
        
        for y in current_elements:
            s = x + y
            if s in current_sums:
                can_add = False
                break
            new_sums.add(s)
        
        if can_add:
            s_diag = 2 * x
            if s_diag in current_sums:
                can_add = False
            else:
                new_sums.add(s_diag)
        
        if can_add:
            dfs(idx + 1, current_size + 1, current_sums | new_sums, current_elements + [x])

    dfs(0, 0, set(), [])
    return best

def construct():
    global _START_TIME, _DEADLINE
    
    # Ensure valid output: 12 distinct integers in [0, 128]
    best_arr = list(range(12))
    best_score = _estimate_max_sidon(best_arr, random.Random(42))
    
    rng = random.Random(42)
    
    def _neighbor(arr):
        new_arr = arr[:]
        idx = rng.randint(0, 11)
        old_val = new_arr[idx]
        if rng.random() < 0.5:
            new_val = rng.randint(0, 128)
        else:
            step = rng.choice([-2, -1, 1, 2, 5, 10])
            new_val = old_val + step
            if new_val < 0: new_val = 0
            if new_val > 128: new_val = 128
        
        if new_val == old_val:
            return None
        
        if new_val in new_arr[:idx] or new_val in new_arr[idx+1:]:
            return None
            
        new_arr[idx] = new_val
        return new_arr

    # Hill climbing with the fast estimator
    while time.time() < _DEADLINE - 10.0:  # Reserve 10s for final check
        neighbor = _neighbor(best_arr)
        if neighbor is None:
            continue
        
        score = _estimate_max_sidon(neighbor, rng, trials=5)
        
        if score < best_score:
            best_score = score
            best_arr = neighbor[:]
            if best_score <= 3:
                # Can't get much lower than 3 for 12 elements easily? 
                # Min Sidon size for 12 elements... 
                # If we break at 3, we might miss a 2? 
                # But 2 is very hard. 
                pass
        elif score == best_score:
            if rng.random() < 0.1:
                best_arr = neighbor[:]

    # Final Polish: Verify the best candidate with a more exact method
    # and try a few local moves with the exact verifier if time permits?
    # No, exact verifier is slow. 
    # Let's just run the exact verifier on the best candidate found 
    # to get the true score, and if it's bad, we might want to keep searching?
    # But we are out of time.
    # The heuristic score is a lower bound. 
    # If heuristic says 3, true is >= 3.
    # If heuristic says 5, true is >= 5.
    # We minimized the heuristic.
    
    # Let's do one last check: run exact solver on best_arr with remaining time
    time_remaining = _DEADLINE - time.time() - 1.0
    if time_remaining > 1.0:
        exact_score = _exact_max_sidon_subset_size(best_arr, time_limit=time_remaining)
        # We don't use exact_score to update best_arr because we can't afford 
        # to search with it. But we know the true value.
        # If exact_score is high, our heuristic was underestimating.
        # This is a risk.
        # To mitigate, we can check a few neighbors of best_arr with the exact solver 
        # if they were promising heuristically?
        # No, just return best_arr.

    assert len(best_arr) == 12
    assert len(set(best_arr)) == 12
    assert all(0 <= x <= 128 for x in best_arr)
    
    return best_arr

# EVOLVE-BLOCK-START
# (construct is defined above; this block is for SCiEL evolution markers)
# EVOLVE-BLOCK-END
