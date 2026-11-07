"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

DEADLINE = 85.0  # safety margin before 90s kill

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i."""
    # For each i in 1..n, find positions j in 0..L-1 where (m+1+j) % i == 0
    # i.e., j % i == (-(m+1)) % i
    options = []
    for i in range(1, n + 1):
        target = (-(m + 1)) % i
        mask = 0
        j = target
        while j < L:
            mask |= (1 << j)
            j += i
        if mask == 0:
            return False
        options.append((mask.bit_count(), mask))
    
    # Most constrained first
    options.sort(key=lambda x: x[0])
    
    n_opts = n
    
    def dfs(k, used_mask):
        if k == n_opts:
            return True
        _, mask = options[k]
        avail = mask & ~used_mask
        # Try each available position
        while avail:
            low = avail & (-avail)
            avail ^= low
            if dfs(k + 1, used_mask | low):
                return True
        return False
    
    return dfs(0, 0)

def _f_val(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L. Lower bound: n (need at least n distinct values).
    # Upper bound: something reasonable. For n=12, L is at most ~144.
    lo = n
    hi = 200  # safe upper bound
    # First check if lo works
    if not _can_match(m, lo, n):
        while lo < hi:
            mid = (lo + hi) // 2
            if _can_match(m, mid, n):
                hi = mid
            else:
                lo = mid + 1
        if lo == hi and _can_match(m, lo, n):
            return lo
        return lo  # may not be valid but we handle this
    return lo

def construct():
    start_time = time.time()
    best_m = 12
    best_L = _f_val(12)
    
    # Search over candidate m values
    # Strategy: try many m values, evaluate f(12, m), keep the best
    # To be efficient, we can skip m values that are likely bad.
    # Key insight: f(12, m) depends on m mod lcm(1..12) = 27720
    # So f(12, m) = f(12, m mod 27720) for m >= some threshold? Not exactly,
    # because the interval (m, m+L] shifts. But the pattern of divisibility
    # repeats with period 27720.
    
    # Actually, f(12, m) is periodic in m with period 27720? Let's think:
    # The condition is about (m, m+L] containing values divisible by i.
    # If we shift m by 27720, the divisibility pattern of the numbers in the
    # interval shifts, but the relative pattern within any window of length L
    # is the same. However, the absolute positions matter for which specific
    # numbers are present.
    
    # Better approach: just search m in [0, 1000000] with some heuristics.
    # We have ~80 seconds. Let's estimate how many m values we can check.
    # Each _f_val call involves binary search (~8 iterations) each with a DFS.
    # The DFS for n=12 with L~100 could be expensive. Let's be careful.
    
    # Let's first try a random/sample search to find good candidates,
    # then refine around them.
    
    import random
    random.seed(42)
    
    # Phase 1: Sample a spread of m values
    candidates = list(range(0, 1000001, 1000))  # 1000 candidates
    # Add some random ones
    for _ in range(500):
        candidates.append(random.randint(0, 1000000))
    
    # Sort candidates for deterministic behavior
    # But let's interleave: check evenly spaced first, then random
    
    checked = set()
    
    def try_m(m):
        nonlocal best_m, best_L
        if m in checked:
            return
        checked.add(m)
        if time.time() - start_time > DEADLINE:
            return
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
    
    # Phase 1: Evenly spaced
    for m in range(0, 1000001, 1000):
        if time.time() - start_time > DEADLINE:
            break
        try_m(m)
    
    # Phase 2: Random samples
    for _ in range(500):
        if time.time() - start_time > DEADLINE:
            break
        m = random.randint(0, 1000000)
        try_m(m)
    
    # Phase 3: Local search around best
    if time.time() - start_time < DEADLINE - 10:
        for offset in range(-50, 51):
            if time.time() - start_time > DEADLINE:
                break
            m = best_m + offset
            if 0 <= m <= 1000000:
                try_m(m)
    
    # Phase 4: More local search if time permits
    if time.time() - start_time < DEADLINE - 5:
        for offset in range(-200, 201):
            if time.time() - start_time > DEADLINE:
                break
            m = best_m + offset
            if 0 <= m <= 1000000:
                try_m(m)
    
    # Phase 5: Broader search around best
    if time.time() - start_time < DEADLINE - 3:
        for offset in range(-1000, 1001, 10):
            if time.time() - start_time > DEADLINE:
                break
            m = best_m + offset
            if 0 <= m <= 1000000:
                try_m(m)
    
    return int(best_m)
# EVOLVE-BLOCK-END
