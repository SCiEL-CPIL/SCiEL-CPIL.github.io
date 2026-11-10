# EVOLVE-BLOCK-START
import time

def _is_valid_set(S):
    """Check if set S satisfies the no-mean-of-subset property."""
    S_list = sorted(S)
    n = len(S_list)
    if n <= 2:
        return True
    # For each element a in S, check if any subset of S\{a} has sum = k*a for k>=2
    max_sum = sum(S_list)
    for i, a in enumerate(S_list):
        others = [x for j, x in enumerate(S_list) if j != i]
        if len(others) < 2:
            continue
        # Subset sum DP using bitset
        # Bit i set means sum i is achievable
        bits = 1  # sum 0 is achievable
        for x in others:
            bits |= (bits << x)
        # Check if k*a is achievable for k = 2, 3, ...
        # Max k: k*a <= sum(others), so k <= sum(others)//a
        max_k = sum(others) // a
        for k in range(2, max_k + 1):
            target = k * a
            if target > max_sum:
                break
            if (bits >> target) & 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        candidate = selected + [x]
        if _is_valid_set(candidate):
            selected = candidate
            if len(selected) >= 48:
                break
    return selected
# EVOLVE-BLOCK-END
