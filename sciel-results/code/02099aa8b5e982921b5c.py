import time

def _is_valid(arr):
    """Check: distinct ints in 1..256, size 1..60, cardinality-separated subset sums."""
    n = len(arr)
    if n < 1 or n > 60:
        return False
    s = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in s:
            return False
        s.add(v)
    # dp[k] is a bitset: bit j set means sum j is achievable with exactly k elements
    max_sum = sum(arr)
    if max_sum > 256 * 60:  # safety
        return False
    dp = [0] * (n + 1)
    dp[0] = 1  # empty subset, sum 0
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Check no sum appears in two different cardinalities
    seen = 0
    for k in range(n + 1):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _try_add(arr, x):
    """Try adding x to arr, return new arr if valid, else None."""
    if x in arr:
        return None
    new_arr = arr + [x]
    if _is_valid(new_arr):
        return new_arr
    return None

def _grow(arr, candidates, deadline):
    """Greedily grow arr by trying candidates, return best valid set."""
    best = arr
    best_len = len(arr)
    # Sort candidates to try promising ones first
    for x in candidates:
        if time.time() > deadline:
            break
        new_arr = _try_add(best, x)
        if new_arr is not None:
            best = new_arr
            best_len += 1
            if best_len >= 60:
                break
    return best

def construct():
    deadline = time.time() + 80.0  # 80s safety margin for 90s limit
    
    # Strategy: use multiple starting points and greedy growth
    # Key insight: elements that are powers of 2 or near-powers work well
    # because subset sums are spread out
    
    best_result = None
    best_size = 0
    
    # Try different starting strategies
    start_strategies = [
        # Powers of 2 and near-powers (good separation)
        [1, 2, 4, 8, 16, 32, 64, 128],
        # Larger base
        [2, 4, 8, 16, 32, 64, 128],
        # Spread out
        [1, 3, 7, 15, 31, 63, 127],
        # Even spread
        [2, 6, 12, 24, 48, 96],
        # Start with larger values
        [10, 20, 40, 80, 160],
        # Mixed
        [1, 2, 5, 10, 20, 40, 80, 160],
    ]
    
    all_candidates = list(range(1, 257))
    
    for start in start_strategies:
        if time.time() > deadline:
            break
        # Verify start is valid
        if not _is_valid(start):
            continue
        
        arr = list(start)
        
        # Multiple passes of greedy growth with different orderings
        for pass_num in range(5):
            if time.time() > deadline:
                break
            
            if pass_num == 0:
                candidates = all_candidates
            elif pass_num == 1:
                candidates = sorted(all_candidates, reverse=True)
            elif pass_num == 2:
                candidates = sorted(all_candidates, key=lambda x: x % 13)
            elif pass_num == 3:
                candidates = [x for x in all_candidates if x % 2 == 0] + \
                             [x for x in all_candidates if x % 2 == 1]
            else:
                candidates = sorted(all_candidates, key=lambda x: x.bit_length())
            
            arr = _grow(arr, candidates, deadline)
            
            if len(arr) > best_size:
                best_size = len(arr)
                best_result = arr
        
        # Also try: remove one element and add two different ones
        if time.time() > deadline:
            break
        for i in range(len(arr)):
            if time.time() > deadline:
                break
            reduced = arr[:i] + arr[i+1:]
            if not _is_valid(reduced):
                continue
            # Try to add two elements
            for x in all_candidates:
                if time.time() > deadline:
                    break
                if x in arr:
                    continue
                new1 = _try_add(reduced, x)
                if new1 is None:
                    continue
                for y in all_candidates:
                    if time.time() > deadline:
                        break
                    if y in arr or y == x:
                        continue
                    new2 = _try_add(new1, y)
                    if new2 is not None and len(new2) > best_size:
                        best_size = len(new2)
                        best_result = new2
    
    if best_result is None:
        # Fallback: return a known valid small set
        return list(range(1, 9))
    
    return best_result
