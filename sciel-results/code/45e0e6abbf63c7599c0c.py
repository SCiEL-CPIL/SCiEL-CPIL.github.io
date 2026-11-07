import time
import itertools
import sys

# Use exact bitset DP for cardinality-indexed subset sums
# dp[k] is a bitset where bit s is set iff there exists a subset of cardinality k summing to s

def _verify(arr):
    n = len(arr)
    if n < 1 or n > 60:
        return False
    seen = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in seen:
            return False
        seen.add(v)
    # dp[k]: bitset of achievable sums with exactly k elements
    max_sum = sum(arr)
    dp = [1] * (n + 1)  # dp[k] starts with sum 0 achievable for k=0
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Check that sums for different cardinalities are disjoint
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _try_add(arr, v, dp_cache):
    """Try adding v to arr; return True if valid, and optionally return updated dp.
    Uses incremental DP: build dp for arr once, then layer v on top."""
    n = len(arr)
    max_n = n + 1
    # Build dp for arr
    dp = [0] * (max_n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Now add v (incremental layer)
    for k in range(max_n, 0, -1):
        dp[k] |= (dp[k - 1] << v)
    # Check disjointness
    acc = 0
    for k in range(max_n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _try_swap(arr, remove_idx, add_v, dp_cache):
    """Try replacing arr[remove_idx] with add_v; return True if the new set is valid."""
    n = len(arr)
    # Build dp for arr without the removed element
    dp = [0] * (n + 1)
    dp[0] = 1
    for i, x in enumerate(arr):
        if i == remove_idx:
            continue
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Now add v
    for k in range(n, 0, -1):
        dp[k] |= (dp[k - 1] << add_v)
    # Check disjointness
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True


def _grow_greedy(start, deadline, max_time_per_step=0.5):
    """Greedy growth: repeatedly add the smallest valid element."""
    cur = list(start)
    cur_set = set(cur)
    improved = True

    while improved and time.monotonic() < deadline:
        improved = False
        for v in range(1, 257):
            if v in cur_set:
                continue
            if _try_add(cur, v, None):
                cur.append(v)
                cur_set.add(v)
                improved = True
                break
        # If we didn't improve, check time
        if not improved:
            break
    return cur


def _swap_search(arr, deadline, max_rounds=20):
    """Local search: try 1-for-1 swaps to escape greedy local optima.
    After each successful swap, attempt greedy regrowth. Returns the best set found."""
    cur = list(arr)
    best = list(arr)
    best_len = len(arr)

    for rnd in range(max_rounds):
        if time.monotonic() > deadline:
            break
        cur_set = set(cur)
        swapped = False
        # Try each position for removal
        for i in range(len(cur)):
            if time.monotonic() > deadline:
                break
            old_v = cur[i]
            # Try adding various values in place of old_v
            # Prefer values not already in the set, spread out
            candidates = []
            for v in range(1, 257):
                if v == old_v or v in cur_set:
                    continue
                candidates.append(v)
            # Shuffle for diversity
            import random
            rng = random.Random(hash((tuple(sorted(cur)), i, rnd)) & 0xFFFFFFFF)
            rng.shuffle(candidates)
            # Limit candidates per position to control cost
            for v in candidates[:30]:
                if time.monotonic() > deadline:
                    break
                if _try_swap(cur, i, v, None):
                    # Perform the swap
                    cur[i] = v
                    cur_set.discard(old_v)
                    cur_set.add(v)
                    swapped = True
                    # Regrow greedily
                    grown = _grow_greedy(cur, deadline)
                    if len(grown) > best_len:
                        best = list(grown)
                        best_len = len(grown)
                    cur = list(grown)
                    break
            if swapped:
                break
        if not swapped:
            break
    return best


def construct():
    t0 = time.monotonic()
    deadline = t0 + 75.0  # safety margin under 90s
    
    best = None
    
    # Strategy 1: Try to build a large set using a structured approach
    # The key insight: if we use elements that are powers of 2 or have large gaps,
    # subset sums for different cardinalities will be naturally separated.
    # 
    # Let's try: use a base set where each element is much larger than the sum of
    # all smaller elements chosen. This ensures unique subset sums overall, which
    # is a stronger condition than needed (which only requires separation by cardinality).
    #
    # For example, if a_i > sum(a_1...a_{i-1}), then all subset sums are unique.
    # Let's greedily build such a set within 1..256.
    
    # Approach: Start with small elements, then jump to large ones
    # Actually, let's try a different structured family:
    # a_1, a_1 + delta, a_1 + 2*delta, ... with appropriate delta
    # This might not work well.
    
    # Better approach: Use the fact that if we pick elements from a geometric-ish sequence,
    # we can ensure the property. Let's try to maximize the size.
    
    # Let's first try consecutive ranges from the high end
    for n in range(60, 10, -1):
        a = 257 - n
        arr = list(range(a, 257))
        if time.monotonic() > deadline - 2:
            break
        if _verify(arr):
            if best is None or len(arr) > len(best):
                best = arr
    
    # Strategy 2: Greedy growth from best found
    if best is not None:
        g = _grow_greedy(best, deadline)
        if len(g) > len(best):
            best = g
        if time.monotonic() > deadline - 2:
            return best if best is not None else list(range(226, 257))

    # Strategy 2b: Local search with swaps to escape greedy local optima
    if best is not None and time.monotonic() < deadline - 5:
        s = _swap_search(best, deadline, max_rounds=10)
        if len(s) > len(best):
            best = s
        if time.monotonic() > deadline - 2:
            return best
    
    # Strategy 3: Try to build from scratch with a smarter heuristic
    # Use a two-phase approach: pick a "core" of small numbers, then add large numbers
    # that won't cause cardinality collisions.
    
    # Try various structured starts
    import random
    rng = random.Random(42)
    
    # Try starting with a small set and growing
    small_starts = []
    # Start with just a few well-chosen elements
    for a1 in [1, 2, 3, 5, 8]:
        for a2 in range(a1 + 1, min(a1 + 20, 20)):
            start = [a1, a2]
            if _verify(start):
                small_starts.append(start)
    
    # Also try geometric-ish starts
    for ratio in [2, 3, 4, 5]:
        arr = [1]
        while arr[-1] * ratio <= 256:
            arr.append(arr[-1] * ratio)
        if len(arr) >= 3 and _verify(arr):
            small_starts.append(arr)
    
    # Grow each start
    for start in small_starts[:20]:  # limit to avoid too many
        if time.monotonic() > deadline - 8:
            break
        g = _grow_greedy(start, min(deadline, t0 + 80))
        if best is None or len(g) > len(best):
            best = g
            # Apply swap search on improved best
            if time.monotonic() < deadline - 5:
                s = _swap_search(best, deadline, max_rounds=5)
                if len(s) > len(best):
                    best = s
    
    # Strategy 4: Random restarts with growth + swap
    attempts = 0
    while time.monotonic() < deadline - 3 and attempts < 300:
        attempts += 1
        n = rng.randint(15, 45)
        # Sample from the full range with some bias toward spread
        if rng.random() < 0.5:
            # Spread out
            step = 256 // n
            base = list(range(1, 257, max(step, 2)))
            if len(base) >= n:
                pool = rng.sample(base, n)
            else:
                pool = rng.sample(range(1, 257), n)
        else:
            pool = rng.sample(range(1, 257), n)
        
        if _verify(pool):
            g = _grow_greedy(pool, min(deadline, t0 + 80))
            if best is None or len(g) > len(best):
                best = g
                # Swap search on the improved best
                if time.monotonic() < deadline - 5:
                    s = _swap_search(best, deadline, max_rounds=5)
                    if len(s) > len(best):
                        best = s
            if best is not None and len(best) >= 60:
                break
    
    # Fallback: a known valid set
    if best is None:
        # Try a simple valid set: powers of 2 (but limited by range)
        best = [1, 2, 4, 8, 16, 32, 64, 128]
        # Verify it works
        if not _verify(best):
            # Very conservative fallback
            best = list(range(129, 193))  # 64 consecutive, likely valid
            if not _verify(best):
                best = list(range(193, 257))  # 64 elements

    # Final: ensure Python ints and cap at 60
    best = [int(v) for v in best]
    if len(best) > 60:
        best = best[:60]
    return best
