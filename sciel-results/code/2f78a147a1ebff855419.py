import time
import random

def _init_dp(arr):
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _verify_add(dp, n, x):
    """
    Checks if adding x to a set of size n (represented by dp) is valid.
    Returns new_dp if valid, None otherwise.
    """
    new_dp = [0] * (n + 2)
    for k in range(n + 1):
        new_dp[k] = dp[k]
    
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << x)
        
    acc = 0
    for k in range(n + 2):
        if new_dp[k] & acc:
            return None
        acc |= new_dp[k]
        
    return new_dp

def _grow_greedy(start, strategy='smallest', deadline=None, rng=None):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = _init_dp(cur)
    
    improved = True
    while improved:
        if deadline is not None and time.monotonic() > deadline:
            break
        improved = False
        
        # Get candidates
        candidates = [v for v in range(1, 257) if v not in cur_set]
        if not candidates:
            break
            
        if strategy == 'smallest':
            order = candidates
        elif strategy == 'largest':
            order = list(reversed(candidates))
        elif strategy == 'random':
            order = candidates[:]
            if rng:
                rng.shuffle(order)
            else:
                random.shuffle(order)
        else:
            order = candidates
            
        for v in order:
            new_dp = _verify_add(dp, n, v)
            if new_dp is not None:
                cur.append(v)
                cur_set.add(v)
                n += 1
                dp = new_dp
                improved = True
                break
                
    return cur

def _try_insertion(cur, dp, n, v):
    """Try to insert v into cur, returning (new_cur, new_dp) or (None, None)"""
    new_dp = _verify_add(dp, n, v)
    if new_dp is not None:
        return cur + [v], new_dp
    return None, None

def _polish(cur, deadline):
    """Try to improve the set by swapping elements"""
    cur = list(cur)
    n = len(cur)
    dp = _init_dp(cur)
    cur_set = set(cur)
    
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        # Try replacing each element with a smaller unused value
        for i in range(n):
            old = cur[i]
            remaining = cur[:i] + cur[i+1:]
            remaining_set = set(remaining)
            rem_dp = _init_dp(remaining)
            
            for v in range(1, old):
                if v in remaining_set:
                    continue
                new_dp = _verify_add(rem_dp, n - 1, v)
                if new_dp is not None:
                    # Also try to add back old or larger values
                    cur_new = remaining + [v]
                    cur_new.sort()
                    dp_new = _init_dp(cur_new)
                    
                    # Try to grow from here
                    cur_grown = _grow_greedy(cur_new, 'smallest', deadline)
                    if len(cur_grown) > len(cur):
                        cur = cur_grown
                        n = len(cur)
                        dp = _init_dp(cur)
                        cur_set = set(cur)
                        improved = True
                        break
            if improved:
                break
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0  # Safety margin
    best = None
    
    def update_best(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            best = list(arr)
    
    # 1. Deterministic Greedies
    # Strategy: Smallest
    res = _grow_greedy([], 'smallest', deadline)
    update_best(res)
    
    # Strategy: Largest
    res = _grow_greedy([], 'largest', deadline)
    update_best(res)
    
    # Start from Powers of 2
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    res = _grow_greedy(p2, 'smallest', deadline)
    update_best(res)
    res = _grow_greedy(p2, 'largest', deadline)
    update_best(res)
    
    # Start from small consecutive sets
    for start_size in [1, 2, 3]:
        for start in range(1, 257 - start_size + 1):
            if time.monotonic() > deadline:
                break
            initial = list(range(start, start + start_size))
            res = _grow_greedy(initial, 'smallest', deadline)
            update_best(res)
    
    # 2. Randomized Restarts with Greedy Growth
    rng = random.Random(42)
    if time.monotonic() < deadline:
        # Try random starting sets with random growth
        for attempt in range(200):
            if time.monotonic() > deadline - 2:
                break
            # Generate a random small starting set
            k = rng.randint(0, 5)
            if k == 0:
                start = []
            else:
                start = rng.sample(range(1, 257), k)
                # Verify start is valid
                start_dp = _init_dp(start)
                acc = 0
                valid = True
                for k_val in range(len(start) + 1):
                    if start_dp[k_val] & acc:
                        valid = False
                        break
                    acc |= start_dp[k_val]
                if not valid:
                    continue
            
            # Try different strategies
            strategy = rng.choice(['smallest', 'largest', 'random'])
            res = _grow_greedy(start, strategy, deadline, rng)
            update_best(res)
    
    # 3. Polish the best solution
    if best is not None and time.monotonic() < deadline - 5:
        polished = _polish(best, deadline)
        update_best(polished)
    
    # If best is None (should not happen), fallback
    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best
