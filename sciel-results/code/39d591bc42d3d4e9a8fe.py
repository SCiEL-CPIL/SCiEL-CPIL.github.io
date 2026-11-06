# EVOLVE-BLOCK-START
import time
import random

def _all_subset_sums_distinct(arr):
    """Check that all 2^n subset sums are distinct."""
    n = len(arr)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    return True

def _get_in_out_sums(arr, i):
    """
    Returns (S_in, S_out) where S_in are subset sums including arr[i]
    and S_out are subset sums excluding arr[i].
    """
    n = len(arr)
    S_in = set()
    S_out = set()
    for mask in range(1 << n):
        s = 0
        for j in range(n):
            if mask & (1 << j):
                s += arr[j]
        if mask & (1 << i):
            S_in.add(s)
        else:
            S_out.add(s)
    return S_in, S_out

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Start with powers of 2
    arr = [1 << i for i in range(12)]
    best_max = max(arr)
    best_arr = list(arr)
    
    # Verify initial solution is valid
    assert _all_subset_sums_distinct(best_arr)
    
    random.seed(42)
    
    # Precompute S_in and S_out for the current state to speed up checks
    # We will update these incrementally or recompute when necessary.
    # Given n=12, recomputing S_in/S_out takes ~12*4096 ops, which is fast.
    # But we can do better.
    
    # Strategy: Local search with fast collision check.
    # For each index i, try to reduce arr[i].
    # Check: (S_in - delta) is disjoint from S_out.
    
    improved = True
    # We will use a more aggressive search with the fast check.
    # To allow larger jumps, we can try random deltas or a range of deltas.
    
    # Initialize S_in, S_out for i=11 (largest element)
    # Actually, we can just recompute S_in/S_out for the current i inside the loop if needed,
    # or maintain them. Since n is small, recomputing is fine if the check is the bottleneck.
    # Wait, the check itself is O(2^{n-1}). Recomputing S_in/S_out is O(n 2^n).
    # It's better to compute S_in/S_out once per i-change attempt setup.
    
    while improved and time.time() < deadline:
        improved = False
        # Try reducing each element
        for i in range(11, -1, -1):
            if time.time() > deadline:
                break
            
            old_val = arr[i]
            if old_val <= 1:
                continue
            
            # Compute S_in and S_out for current state
            S_in, S_out = _get_in_out_sums(arr, i)
            
            # We want to find the largest delta > 0 such that
            # 1. arr[i] - delta > 0
            # 2. arr[i] - delta not in {arr[j] for j != i}
            # 3. (S_in - delta) is disjoint from S_out
            
            # Since we want to minimize max, we want to reduce arr[i] as much as possible.
            # We can binary search for the max valid delta, or just try large deltas.
            # Let's try a binary search on delta from 1 to old_val - 1.
            
            low = 1
            high = old_val - 1
            best_delta = 0
            
            # Check if delta is valid
            def is_valid_delta(d):
                new_val = old_val - d
                if new_val <= 0:
                    return False
                if any(arr[j] == new_val for j in range(12) if j != i):
                    return False
                # Check collision: (S_in - d) ∩ S_out == ∅
                # This is equivalent to: no s in S_in such that s - d in S_out
                # Or: no s in S_out such that s + d in S_in
                # Using set intersection is efficient.
                # shifted_S_in = {s - d for s in S_in}
                # return shifted_S_in.isdisjoint(S_out)
                # To avoid creating a new set, we can iterate.
                # But 2048 elements is small.
                for s in S_in:
                    if s - d in S_out:
                        return False
                return True
            
            # Binary search for max valid delta
            # Note: if delta is valid, smaller deltas might not be valid?
            # Actually, if delta is valid, it doesn't imply delta-1 is valid.
            # The condition is not monotonic in delta.
            # Example: S_in={10}, S_out={9, 11}.
            # d=1: 10-1=9 (in S_out) -> Invalid
            # d=2: 10-2=8 (not in S_out) -> Valid
            # d=3: 10-3=7 (not in S_out) -> Valid
            # So we can't binary search directly for "max valid".
            # We should scan or use a smarter approach.
            # However, we want to reduce as much as possible.
            # Let's try large deltas first.
            
            # Try deltas from high to low? That's slow if high is large.
            # But old_val is at most 2048 initially.
            # Let's try a heuristic: try delta = old_val // 2, then adjust.
            # Or just iterate delta from 1 to old_val-1? 2048 iterations * 2048 check = 4M ops per i.
            # 12 elements * 4M = 48M ops per pass.
            # 60s / 48M is plenty of time for many passes.
            
            # Let's optimize the check loop.
            # We can precompute S_out as a set.
            # The check `s - d in S_out` is O(1).
            
            # To find the *maximum* reduction, we can iterate d from 1 to max_d.
            # But we want the *largest* d that is valid.
            # Let's just iterate d from 1 to old_val-1 and keep the best one found?
            # No, we want to apply the best one.
            # Let's find ALL valid deltas and pick the largest.
            # This is O(old_val * |S_in|).
            
            # Alternative: Just try a few large random deltas or a specific sequence.
            # Given the time budget, a simple scan for the best delta is robust.
            
            best_delta_found = 0
            # We can speed this up by checking only a subset of deltas if we are stuck,
            # but let's start with a full scan for the first few iterations to find a good move.
            
            # Optimization: If we find a valid delta, we can stop if we're just looking for *any* improvement?
            # No, we want to minimize the max, so we want the *largest* valid delta for this element.
            
            # Let's try to jump to a value that is significantly smaller.
            # We'll check deltas in descending order? No, that's expensive if most are invalid.
            # Let's check deltas in ascending order and keep track of the largest valid one.
            
            for d in range(1, old_val):
                if is_valid_delta(d):
                    best_delta_found = d # We want the largest, so we keep updating
                    # We can't break because a larger d might be valid.
            
            if best_delta_found > 0:
                arr[i] = old_val - best_delta_found
                improved = True
                if max(arr) < best_max:
                    best_max = max(arr)
                    best_arr = list(arr)
    
    # Final check
    if _all_subset_sums_distinct(best_arr):
        return best_arr
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
