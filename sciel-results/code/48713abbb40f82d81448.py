"""Stronger algorithm: greedy construction with fast validity check."""
import time

def _fast_check_valid(s):
    """
    Check if set s has cardinality-separated subset sums.
    Returns True if valid, False otherwise.
    Uses DP: sums_by_card[k] = set of all subset sums using exactly k elements.
    For validity, we need sums_by_card[k] ∩ sums_by_card[j] = ∅ for all k ≠ j.
    """
    n = len(s)
    # sums_by_card[k] will be a set of ints
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for x in s:
        # Update from high k to low k to avoid using x multiple times
        for k in range(n, 0, -1):
            if sums_by_card[k - 1]:
                sums_by_card[k].update(v + x for v in sums_by_card[k - 1])
    
    # Check for collisions between different cardinalities
    for k in range(n + 1):
        for j in range(k + 1, n + 1):
            if sums_by_card[k] & sums_by_card[j]:
                return False
    return True

def _greedy_build(max_size=60, max_val=256, time_limit=50.0):
    """
    Greedy construction: start with empty set, try adding candidate values
    in descending order of "safety" (values that are far from causing collisions).
    Use a randomized restart strategy.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_set = [256]  # fallback: single element always valid
    best_size = 1
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Randomize the candidate order slightly for diversity
        # Use a random permutation of candidate values, but bias towards larger values
        # Larger values tend to create more separated sums
        
        # Candidate values: 1 to 256
        # We'll try adding them in a random order, but weighted towards larger values
        import random
        candidates = list(range(1, max_val + 1))
        
        # Weight towards larger values for the initial construction
        random.shuffle(candidates)
        
        current_set = []
        # Precompute sums_by_card for fast incremental updates
        # We'll maintain this incrementally
        
        for x in candidates:
            if len(current_set) >= max_size:
                break
            if time.time() > deadline - 5:
                break
            
            # Try adding x to current_set
            new_set = current_set + [x]
            if _fast_check_valid(new_set):
                current_set = new_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Start with a known valid set
    best_set = list(range(128, 136))  # 8 elements
    best_size = len(best_set)
    
    # Try to build a larger set using greedy approach
    if time.time() < deadline - 10:
        result = _greedy_build(max_size=60, max_val=256, time_limit=70.0)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
    
    # Verify the final result
    if not _fast_check_valid(best_set):
        # Fallback to a definitely valid set
        best_set = [256]
    
    # Ensure we return Python ints
    return [int(x) for x in best_set]
