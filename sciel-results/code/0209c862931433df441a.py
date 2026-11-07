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

def _is_valid_dp(dp):
    acc = 0
    for k in range(len(dp)):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _verify_add(dp, n, x):
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

def _count_collisions(dp):
    """
    Returns the number of pairs of subsets with different cardinalities 
    that have the same sum.
    """
    collisions = 0
    acc = 0
    for k in range(len(dp)):
        # Count bits in common between dp[k] and acc
        common = dp[k] & acc
        # Count number of set bits in common
        collisions += bin(common).count('1')
        acc |= dp[k]
    return collisions

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
        elif strategy == 'scored':
            # Score candidates by how "safe" they are (fewer collisions)
            scored = []
            for v in candidates:
                new_dp = _verify_add(dp, n, v)
                if new_dp is not None:
                    # Count total bits set in new_dp as a "complexity" measure
                    total_bits = sum(bin(x).count('1') for x in new_dp)
                    scored.append((total_bits, v))
            scored.sort()
            order = [v for _, v in scored]
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

def _local_search(cur, deadline, rng):
    """
    Local search with swaps to escape local optima.
    """
    cur = list(cur)
    n = len(cur)
    cur_set = set(cur)
    
    # Try to improve by swapping elements
    for _ in range(1000):
        if time.monotonic() > deadline:
            break
        
        # Try random swaps
        if rng is None:
            rng = random.Random()
            
        swapped = False
        for _ in range(10): # Try a few swaps per iteration
            i = rng.randint(0, n - 1)
            old = cur[i]
            
            # Find a candidate to swap with
            candidates = [v for v in range(1, 257) if v not in cur_set]
            if not candidates:
                break
                
            v = rng.choice(candidates)
            
            # Perform swap
            cur[i] = v
            cur_set.remove(old)
            cur_set.add(v)
            
            # Check validity
            cur_sorted = sorted(cur)
            dp = _init_dp(cur_sorted)
            
            if _is_valid_dp(dp):
                swapped = True
                break
            else:
                # Revert
                cur[i] = old
                cur_set.add(old)
                cur_set.remove(v)
                
        if swapped:
            # If we swapped, try to grow again
            cur = _grow_greedy(cur, 'smallest', deadline, rng)
            n = len(cur)
            cur_set = set(cur)
        else:
            # If no swap worked, try a more aggressive move: remove one and add two?
            # Or just break and restart?
            # Let's try removing a random element and then growing
            if n > 0:
                i = rng.randint(0, n - 1)
                old = cur[i]
                cur.pop(i)
                cur_set.remove(old)
                n -= 1
                # Grow from this smaller set
                cur = _grow_greedy(cur, 'random', deadline, rng)
                n = len(cur)
                cur_set = set(cur)
                
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0
    best = None
    rng = random.Random(42)
    
    def update_best(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            best = list(arr)
            
    # 1. Deterministic Greedies
    for strategy in ['smallest', 'largest', 'scored']:
        if time.monotonic() > deadline:
            break
        res = _grow_greedy([], strategy, deadline, rng)
        update_best(res)
        
    # Start from Powers of 2
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for strategy in ['smallest', 'largest', 'scored']:
        if time.monotonic() > deadline:
            break
        res = _grow_greedy(p2, strategy, deadline, rng)
        update_best(res)
        
    # 2. Randomized Restarts
    if time.monotonic() < deadline:
        for attempt in range(200):
            if time.monotonic() > deadline - 2:
                break
            k = rng.randint(0, 8)
            if k == 0:
                start = []
            else:
                start = rng.sample(range(1, 257), k)
                start.sort()
                start_dp = _init_dp(start)
                if not _is_valid_dp(start_dp):
                    continue
            
            strategy = rng.choice(['smallest', 'largest', 'scored', 'random'])
            res = _grow_greedy(start, strategy, deadline, rng)
            update_best(res)
            
    # 3. Local Search on best
    if best is not None and time.monotonic() < deadline - 10:
        polished = _local_search(best, deadline, rng)
        update_best(polished)
        
    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best
