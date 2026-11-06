import time
import random
import numpy as np

def _sum_counts(mask):
    """Compute sum representation counts for a binary mask over 1..1000.
    mask: bool array of length 1000, mask[i-1] = True means i is in set.
    Returns: array of length 2001 (indices 2..2000) with counts of a+b=a+b' with a<=b.
    """
    idx = np.where(mask)[0]  # 0-indexed positions, so values are idx+1
    if len(idx) == 0:
        return np.zeros(2001, dtype=np.int32)
    # Build all pairs (i, j) with i <= j from selected indices
    # Use outer sum
    vals = idx + 1  # actual values 1..1000
    # We need all pairs (a, b) with a <= b from vals
    # Use broadcasting: for each element, sum with all elements >= it
    # More efficient: compute all pairwise sums, then filter a<=b
    # For n elements, this is n*(n+1)/2 pairs
    n = len(vals)
    if n > 0:
        # Create upper triangular indices
        ii, jj = np.triu_indices(n)
        sums = vals[ii] + vals[jj]
        counts = np.bincount(sums, minlength=2001).astype(np.int32)
        return counts
    return np.zeros(2001, dtype=np.int32)

def _can_add(mask, x, counts):
    """Check if adding element x (1-indexed) to the set keeps all sums <= 2.
    mask: current binary mask
    x: value to add (1..1000)
    counts: current sum counts array
    Returns: True if x can be added.
    """
    if mask[x - 1]:
        return False
    # New sums created: x + y for all y in set (y >= x to avoid double counting? No)
    # Actually, sums x+y where y is in set. Since we count a+b with a<=b,
    # adding x creates sums x+y for all y in set with y >= x, and x+x.
    # But also x+y for y < x was already counted as y+x.
    # Wait, the count array counts all pairs a+b with a<=b from the set.
    # If I add x, the new pairs are (x, y) for all y in set where x <= y, plus (x, x).
    # For y < x, the pair (y, x) is new and adds to sum x+y.
    # So new sums: for each y in set, sum x+y gets +1. Plus x+x gets +1.
    # But x+x is included when y=x... but x is not in set yet.
    # So: new sums are x+y for all y in set (each gets +1), and x+x (gets +1).
    idx = np.where(mask)[0] + 1  # values in set
    if len(idx) > 0:
        new_sums = x + idx  # x + each element in set
        # Check if any of these sums already has count >= 2
        if np.any(counts[new_sums] >= 2):
            return False
    # Also check x + x
    if counts[2 * x] >= 2:
        return False
    return True

def _add_element(mask, x, counts):
    """Add element x to mask and update counts in-place."""
    mask[x - 1] = True
    idx = np.where(mask)[0] + 1  # all values in set including x
    # The new sums are x + y for all y in set (y != x), and x + x
    # Actually simpler: recompute affected sums
    # Sums involving x: x + y for all y in set (including x itself)
    other_vals = idx[idx != x]
    if len(other_vals) > 0:
        new_sums = x + other_vals
        counts[new_sums] += 1
    counts[2 * x] += 1

def _remove_element(mask, x, counts):
    """Remove element x from mask and update counts in-place."""
    if not mask[x - 1]:
        return
    mask[x - 1] = False
    idx = np.where(mask)[0] + 1  # remaining values
    other_vals = idx  # all remaining
    if len(other_vals) > 0:
        new_sums = x + other_vals
        counts[new_sums] -= 1
    counts[2 * x] -= 1

def _greedy_order(order, mask, counts):
    """Process elements in given order, adding if possible."""
    for x in order:
        if _can_add(mask, x, counts):
            _add_element(mask, x, counts)

def _local_search(mask, counts, max_steps=5000, time_limit=None):
    """Local search: try swaps and adds to improve size."""
    start_time = time.time()
    best_size = np.sum(mask)
    best_mask = mask.copy()
    best_counts = counts.copy()
    
    for step in range(max_steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Try adding a random element
        added = False
        for _ in range(20):
            x = random.randint(1, 1000)
            if not mask[x - 1] and _can_add(mask, x, counts):
                _add_element(mask, x, counts)
                added = True
                break
        
        if added:
            if np.sum(mask) > best_size:
                best_size = np.sum(mask)
                best_mask = mask.copy()
                best_counts = counts.copy()
            continue
        
        # Try a swap: remove one, add another
        in_set = np.where(mask)[0] + 1
        if len(in_set) == 0:
            break
        
        removed_x = random.choice(in_set)
        _remove_element(mask, removed_x, counts)
        
        # Try to add a better element
        swapped = False
        for _ in range(30):
            x = random.randint(1, 1000)
            if x != removed_x and not mask[x - 1] and _can_add(mask, x, counts):
                _add_element(mask, x, counts)
                swapped = True
                break
        
        if swapped:
            if np.sum(mask) > best_size:
                best_size = np.sum(mask)
                best_mask = mask.copy()
                best_counts = counts.copy()
        else:
            # Revert the removal
            _add_element(mask, removed_x, counts)
    
    return best_mask, best_counts, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    best_mask = np.zeros(1000, dtype=bool)
    best_size = 0
    
    # Run multiple greedy + local search iterations
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        mask = np.zeros(1000, dtype=bool)
        counts = np.zeros(2001, dtype=np.int32)
        
        # Random order for greedy
        order = list(range(1, 1001))
        random.shuffle(order)
        
        # Greedy phase
        _greedy_order(order, mask, counts)
        
        current_size = np.sum(mask)
        
        # Local search phase with time limit
        remaining = deadline - time.time()
        ls_time = min(remaining * 0.5, 5.0)  # Up to 5s per local search
        if ls_time > 0.1:
            mask, counts, current_size = _local_search(
                mask, counts, 
                max_steps=2000, 
                time_limit=ls_time
            )
        
        if current_size > best_size:
            best_size = current_size
            best_mask = mask.copy()
        
        # Check time
        if time.time() > deadline - 2.0:
            break
    
    # Convert mask to list of integers
    result = np.where(best_mask)[0] + 1
    return list(result)
