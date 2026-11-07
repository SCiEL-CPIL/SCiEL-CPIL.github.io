import time
import sys

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
    # Using Python integers as bitsets
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
            
    # Check disjointness
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def construct():
    t0 = time.monotonic()
    deadline = t0 + 80.0
    
    best = None
    
    # Strategy: Find the largest n such that a consecutive set [a, ..., a+n-1] works.
    # Condition derived: a > max_k (k(n-1-k))
    # And a+n-1 <= 256
    
    for n in range(60, 1, -1):
        if time.monotonic() > deadline - 5:
            break
            
        # Calculate required a
        # max_k k(n-1-k)
        max_term = 0
        for k in range(1, n):
            term = k * (n - 1 - k)
            if term > max_term:
                max_term = term
        
        min_a = max_term + 1
        max_a = 257 - n
        
        if min_a <= max_a:
            # Try the smallest valid a (to leave room for potential other strategies, though we just return one)
            # Actually, larger a is safer for separation?
            # Gap = a + k^2 - k(n-1). Larger a -> larger gap.
            # So we should pick the largest possible a?
            # But we want to maximize n. For a fixed n, any valid a works.
            # Let's pick a = min_a to be conservative? Or max_a?
            # Let's try max_a first, it gives the largest sums, but separation is relative.
            # Actually, separation condition is linear in a.
            # Let's just try a range of a's if needed, but usually one works.
            
            # Try a = min_a
            arr = list(range(min_a, min_a + n))
            if _verify(arr):
                best = arr
                break
            else:
                # Try other a's?
                # The condition a > max_term is sufficient for range separation.
                # If range separation holds, the sets are disjoint.
                # So if min_a works, it should be valid.
                # If _verify fails, it means my range separation logic has a flaw or implementation error.
                # Let's try max_a just in case.
                arr2 = list(range(max_a, max_a + n))
                if _verify(arr2):
                    best = arr2
                    break
    
    # Fallback: known valid set
    if best is None:
        # Consecutive integers from high end
        # n=31, a=226
        best = list(range(226, 257))
        if not _verify(best):
            # Try smaller
            best = list(range(200, 257))
            # Trim to valid
            while not _verify(best) and len(best) > 1:
                best.pop(0)
                
    # Ensure Python ints
    best = [int(v) for v in best]
    
    return best
