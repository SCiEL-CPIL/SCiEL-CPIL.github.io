import time
import random

def _verify(arr):
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

def _greedy_grow(start, deadline):
    """Greedily add elements to maintain cardinality separation."""
    cur = list(start)
    cur_set = set(cur)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for v in range(1, 257):
            if v in cur_set:
                continue
            cand = cur + [v]
            if _verify(cand):
                cur = cand
                cur_set.add(v)
                improved = True
                break
    return cur

def _swap_refine(arr, deadline):
    """Try swapping elements to enable further growth."""
    cur = list(arr)
    cur_set = set(cur)
    best = list(cur)
    
    while time.monotonic() < deadline:
        improved = False
        for i in range(len(cur)):
            for v in range(1, 257):
                if v in cur_set:
                    continue
                # Swap cur[i] with v
                new_arr = cur[:i] + [v] + cur[i+1:]
                if _verify(new_arr):
                    cur = new_arr
                    cur_set.discard(cur[i])
                    cur_set.add(v)
                    improved = True
                    if len(cur) > len(best):
                        best = list(cur)
                    break
            if improved:
                break
        if not improved:
            # Try to grow after swaps
            grown = _greedy_grow(cur, deadline)
            if len(grown) > len(best):
                best = grown
            if len(best) >= 60:
                break
            if time.monotonic() > deadline - 5:
                break
    
    return best

def construct():
    t0 = time.monotonic()
    deadline = t0 + 80.0  # Safety margin under 90s wall-clock kill
    best = None
    
    # Strategy 1: Start with high-range consecutive and grow
    # Try different starting points
    start_candidates = [
        list(range(256, 225, -1)),  # 226..256 (31 elements)
        list(range(256, 220, -1)),  # 221..256 (36 elements)
        list(range(200, 169, -1)),  # 170..200 (31 elements)
        list(range(128, 97, -1)),   # 98..128 (31 elements)
    ]
    
    for start in start_candidates:
        if time.monotonic() > deadline - 10:
            break
        if _verify(start):
            grown = _greedy_grow(start, deadline - 5)
            if best is None or len(grown) > len(best):
                best = grown
            if len(best) >= 60:
                break
    
    # Strategy 2: Random restarts with different distributions
    rng = random.Random(42)
    attempts = 0
    while time.monotonic() < deadline - 15 and attempts < 10000:
        attempts += 1
        n = rng.randint(25, 50)
        
        # Try different sampling strategies
        r = rng.randint(0, 3)
        if r == 0:
            # Uniform
            pool = list(range(1, 257))
        elif r == 1:
            # High bias
            pool = list(range(100, 257))
        elif r == 2:
            # Bimodal
            low = list(range(1, 80))
            high = list(range(180, 257))
            pool = rng.sample(low, n//2) + rng.sample(high, n - n//2)
        else:
            # Gappy: skip middle
            pool = list(range(1, 100)) + list(range(160, 257))
        
        if len(pool) < n:
            continue
        cand = rng.sample(pool, n)
        if _verify(cand):
            grown = _greedy_grow(cand, deadline - 5)
            if best is None or len(grown) > len(best):
                best = grown
            if len(best) >= 60:
                break
    
    # Strategy 3: Swap refinement on best
    if best is not None and time.monotonic() < deadline - 10:
        refined = _swap_refine(best, deadline - 5)
        if len(refined) > len(best):
            best = refined
    
    # Fallback: ensure we have something valid
    if best is None:
        best = list(range(226, 257))  # 31 elements, should be valid
    
    # Final verification and type safety
    best = [int(v) for v in best]
    if not _verify(best):
        # Absolute fallback
        best = list(range(226, 257))
    
    return best
