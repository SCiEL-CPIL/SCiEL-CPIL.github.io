import time
import random
import sys

def _verify(arr):
    """
    Verify if the set satisfies the cardinality-separated subset sum property.
    Uses bitset DP for efficiency.
    """
    n = len(arr)
    if n < 1 or n > 60:
        return False
    
    # Check types and range
    seen = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
        
    # dp[k] is a bitset where bit s is set iff there exists a subset of cardinality k summing to s
    # We use Python integers as bitsets.
    dp = [0] * (n + 1)
    dp[0] = 1  # Sum 0 achievable with 0 elements
    
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
            
    # Check disjointness of sums for different cardinalities
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
        
    return True

def _try_add(arr, v, dp_state):
    """
    Try adding v to arr.
    Returns (success, new_dp_state)
    dp_state: list of bitsets dp[k]
    """
    n = len(arr)
    max_n = n + 1
    new_dp = [0] * (max_n + 1)
    
    # Copy existing dp
    for k in range(n + 1):
        new_dp[k] = dp_state[k]
        
    # Update with v
    # We must iterate backwards to avoid using v multiple times
    for k in range(max_n, 0, -1):
        new_dp[k] |= (new_dp[k - 1] << v)
        
    # Check disjointness
    acc = 0
    for k in range(max_n + 1):
        if new_dp[k] & acc:
            return False, None
        acc |= new_dp[k]
        
    return True, new_dp

def _grow_greedy(start, deadline, max_time_per_step=0.5):
    """
    Grow the set greedily.
    """
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    
    # Initialize DP
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in cur:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
            
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        # Try adding smallest valid element first? Or random?
        # Smallest might be safer for keeping sums low?
        # Actually, we want to fit in 1..256.
        candidates = [v for v in range(1, 257) if v not in cur_set]
        
        # Heuristic: try candidates that minimize the "gap" violation or something?
        # Just try all, pick the first one that works.
        # To allow for larger sets, maybe try larger numbers first?
        # No, let's just iterate.
        
        for v in candidates:
            success, new_dp = _try_add(cur, v, dp)
            if success:
                cur.append(v)
                cur_set.add(v)
                dp = new_dp
                n = len(cur)
                improved = True
                break
                
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0  # Safety margin
    
    best = None
    best_len = 0
    
    # Strategy 1: Greedy from scratch with different orderings
    # Try adding elements in various orders to find good starting sets
    orderings = [
        list(range(1, 257)),  # Low to high
        list(range(256, 0, -1)),  # High to low
        list(range(129, 257)) + list(range(1, 129)),  # High half then low half
        list(range(1, 129)) + list(range(256, 128, -1)),  # Low half then high half
    ]
    
    for ordering in orderings:
        if time.monotonic() > deadline - 5:
            break
        grown = _grow_greedy([], deadline)
        # Actually, we need to seed it properly. Let's just use the ordering as candidates.
        # But _grow_greedy takes a start list. Let's modify the approach.
        # For now, let's just do random search more aggressively.
        break
    
    # Strategy 2: Random search for larger n
    # If best_len < 17, try random sets of size best_len + 1
    rng = random.Random(42)
    
    target_len = best_len + 1 if best is not None else 15
    
    attempts = 0
    while time.monotonic() < deadline - 5 and attempts < 2000:
        attempts += 1
        
        # Generate a random set of size target_len
        # Bias towards tight clusters
        # Pick a base, pick tight offsets
        if rng.random() < 0.5:
            # Tight cluster
            width = rng.randint(target_len, target_len + 5)
            start = rng.randint(1, 257 - width)
            # Pick target_len distinct numbers in [start, start+width]
            pool = rng.sample(range(start, start + width + 1), target_len)
        else:
            # Uniform random
            pool = rng.sample(range(1, 257), target_len)
            
        if _verify(pool):
            # Grow it
            grown = _grow_greedy(pool, deadline)
            if len(grown) > best_len:
                best = grown
                best_len = len(grown)
            target_len = best_len + 1
            
    # Strategy 3: Local search on best found
    if best is not None and time.monotonic() < deadline - 2:
        # Try to swap elements in best to allow adding more?
        # This is complex. Greedy growth should have handled it.
        pass

    if best is None:
        # Fallback: a known small valid set
        best = [1, 2, 4, 8, 16, 32, 64, 128]
        # Verify fallback
        if not _verify(best):
             # Very conservative
             best = list(range(249, 257)) # 8 consecutive
             if not _verify(best):
                 best = [256]
    
    # Ensure Python ints
    best = [int(v) for v in best]
    return best
