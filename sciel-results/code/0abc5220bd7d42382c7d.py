import time
import random
import numpy as np
from collections import defaultdict

def _count_sums_np(arr):
    """Count all pairwise sums a+b with a<=b from arr using NumPy."""
    if not arr:
        return np.zeros(2002, dtype=np.int32)
    a = np.array(arr)
    # Compute all sums
    # We only need a[i] + a[j] for i <= j
    # np.triu_indices
    n = len(a)
    i, j = np.triu_indices(n)
    sums = a[i] + a[j]
    # sums are in range 2..2000
    # bincount returns counts
    # We need to map sums to indices.
    # Max sum is 2000.
    counts = np.bincount(sums, minlength=2002)
    return counts

def _can_add_np(counts, arr, x):
    """Check if x can be added to arr given current counts."""
    # Sums introduced: x + arr, 2*x
    # We need to ensure counts[s] < 2 for all these s.
    # Note: if x is already in arr, this logic is tricky, but we assume x not in arr.
    # Sums: x + a for a in arr, and 2*x.
    
    # Optimization: if 2*x is already >= 2, return False.
    if counts[2*x] >= 2:
        return False
        
    # Check x + a for all a in arr
    # arr is a list, convert to np array if not already? 
    # In local search, we can keep arr as np array for speed.
    
    sums = x + np.array(arr)
    # Check if any count is >= 2
    if np.any(counts[sums] >= 2):
        return False
    return True

def _add_to_counts(counts, arr, x):
    """Update counts array after adding x."""
    counts[2*x] += 1
    sums = x + np.array(arr)
    counts[sums] += 1

def _remove_from_counts(counts, arr, x):
    """Update counts array after removing x."""
    counts[2*x] -= 1
    sums = x + np.array(arr)
    # Be careful: if x is in arr, we only subtract for other elements?
    # The function _count_sums counts a+b for i<=j.
    # If we remove x, we remove sums x+a for all a in arr (including x if i=j case? no, i<j or i=j).
    # Actually, simpler: just recompute or carefully subtract.
    # Since arr contains x, `x + np.array(arr)` includes `2x`.
    # So `counts[2x]` is incremented once in `_add`.
    # In `_remove`, we should decrement `2x` once.
    # And decrement `x+a` for all a in arr *except* if we double count?
    # Let's trace:
    # Add x: adds 2x, and x+a for all a in old_arr.
    # Remove x: subtracts 2x, and x+a for all a in new_arr (which is old_arr - x? No, arr is current state).
    # If arr is the set *before* removal, then `x + np.array(arr)` includes `2x`.
    # So we subtract 1 from `counts[2x]` and 1 from `counts[x+a]` for all a in arr.
    # This correctly removes the contributions of x.
    
    counts[sums] -= 1
    # Note: counts[2x] was incremented twice in the logic above?
    # `_add`: `counts[2x] += 1` AND `sums` includes `2x` (since x is in arr? No, `_add` is called with `arr` NOT containing x).
    # Wait, in `_add_to_counts(counts, arr, x)`, `arr` is the set BEFORE adding x.
    # So `x + np.array(arr)` does NOT include `2x`.
    # So `counts[2x]` is updated separately.
    # In `_remove_from_counts(counts, arr, x)`, `arr` is the set BEFORE removing x (so it contains x).
    # So `x + np.array(arr)` DOES include `2x`.
    # So `counts[sums] -= 1` will decrement `counts[2x]` once.
    # And we also do `counts[2x] -= 1` explicitly?
    # If we do both, we decrement `2x` twice.
    # Let's check `_add`: `arr` does not have x. `sums` does not have `2x`. `counts[2x] += 1` is correct.
    # Let's check `_remove`: `arr` HAS x. `sums` HAS `2x`. `counts[sums] -= 1` decrements `2x` once.
    # So we should NOT do `counts[2x] -= 1` separately in `_remove`.
    # Let's fix `_remove_from_counts`.
    pass

def _greedy_construct_np():
    """Greedy construction using NumPy for checks."""
    counts = np.zeros(2002, dtype=np.int32)
    chosen = []
    for x in range(1, 1001):
        if _can_add_np(counts, chosen, x):
            _add_to_counts(counts, chosen, x)
            chosen.append(x)
    return chosen, counts

def _local_search_np(chosen, counts, deadline):
    """Local search using NumPy for checks."""
    chosen = list(chosen)
    counts = counts.copy()
    candidates = [x for x in range(1, 1001) if x not in set(chosen)]
    
    while time.time() < deadline:
        # Try to remove one and add two? Or just random swaps?
        # Random swap: remove x, try to add y.
        # To increase size, we need remove 1, add 2.
        # This is hard to do randomly efficiently.
        
        # Alternative: Randomly pick a candidate y. If it can be added, add it.
        # This increases size by 1.
        # But we might get stuck.
        
        # Let's try: Randomly pick x to remove.
        if not chosen: break
        i = random.randint(0, len(chosen)-1)
        x = chosen[i]
        
        # Temporarily remove x
        _remove_from_counts_fix(counts, chosen, x)
        chosen.pop(i)
        
        # Try to add two elements from candidates
        # Shuffle candidates
        random.shuffle(candidates)
        added = []
        for y in candidates:
            if y in chosen: continue
            if _can_add_np(counts, chosen, y):
                _add_to_counts(counts, chosen, y)
                chosen.append(y)
                added.append(y)
                if len(added) == 2:
                    break
        
        if len(added) == 2:
            # Success, size increased by 1
            # Update candidates list
            candidates = [c for c in candidates if c not in set(chosen)]
            # Also, x is now available? No, we removed x, but we added 2 others.
            # x is no longer in chosen. Is x a candidate?
            # We should keep x in candidates? Or just rebuild?
            # Rebuilding is O(N). N=1000. Cheap.
            chosen_set = set(chosen)
            candidates = [x for x in range(1, 1001) if x not in chosen_set]
        else:
            # Failed to add 2. Put x back.
            chosen.insert(i, x)
            _add_from_counts_fix(counts, chosen, x) # Need reverse of remove
            # To avoid complex reverse, maybe just recompute counts?
            # Recomputing is O(N^2). If this happens often, it's slow.
            # Let's implement proper inverse.
            
    return chosen

# Let's implement the fix for remove/add logic carefully.
# State: counts, chosen (list)
# _add(counts, chosen, x): chosen does NOT contain x.
#   counts[2x] += 1
#   sums = x + np.array(chosen)
#   counts[sums] += 1
#   chosen.append(x)

# _remove(counts, chosen, x): chosen DOES contain x.
#   We need to subtract contributions of x.
#   Sums contributed by x: 2x, and x+y for all y in chosen (y != x? No, x+y for all y in chosen, including x? 
#   The original count included x+x (once) and x+y (once for each y).
#   So we subtract 1 from counts[2x].
#   And subtract 1 from counts[x+y] for all y in chosen (y != x).
#   Wait, `x + np.array(chosen)` includes `2x`.
#   So if we do `counts[x + np.array(chosen)] -= 1`, we subtract 1 from `2x` and 1 from `x+y` for all y.
#   This is exactly correct.
#   So `_remove` is just:
#   sums = x + np.array(chosen)
#   counts[sums] -= 1
#   chosen.remove(x)

# Let's verify `_add` again.
# `chosen` does not have x.
# `sums = x + np.array(chosen)` does not have `2x`.
# So we need `counts[2x] += 1` separately.
# And `counts[sums] += 1`.
# Correct.

# So:
def _add_to_counts(counts, chosen, x):
    counts[2*x] += 1
    if chosen:
        sums = x + np.array(chosen)
        counts[sums] += 1
    # chosen.append(x) is done by caller

def _remove_from_counts(counts, chosen, x):
    # chosen contains x
    sums = x + np.array(chosen)
    counts[sums] -= 1
    # chosen.remove(x) is done by caller

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    chosen, counts = _greedy_construct_np()
    
    if time.time() < deadline:
        chosen = _local_search_np(chosen, counts, deadline)
        
    return chosen
