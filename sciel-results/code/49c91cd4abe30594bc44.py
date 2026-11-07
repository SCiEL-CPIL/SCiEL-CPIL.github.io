"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

N = 12
MAX_M = 1000000
MAX_L = 144  # f(12, m) is at most 144 per task description

def _can_match(m, L):
    """
    Check if (m, m+L] contains distinct b_1...b_12 with i | b_i.
    Uses bitmask DFS with most-constrained-first ordering.
    """
    n = N
    # For each index i (1..12), find which positions j in 0..L-1 work
    # Position j corresponds to value v = m + 1 + j
    # We need v % i == 0, i.e., (m + 1 + j) % i == 0
    # => j % i == (-(m+1)) % i
    
    options = []
    for i in range(1, n + 1):
        target = (-(m + 1)) % i
        mask = 0
        for j in range(L):
            if j % i == target:
                mask |= (1 << j)
        if mask == 0:
            return False
        options.append((mask.bit_count(), mask))
    
    # Sort by number of options (most constrained first)
    options.sort(key=lambda x: x[0])
    
    def dfs(k, used_mask):
        if k == n:
            return True
        _, mask = options[k]
        available = mask & ~used_mask
        while available:
            # Pick the lowest set bit
            lowbit = available & (-available)
            if dfs(k + 1, used_mask | lowbit):
                return True
            available ^= lowbit
        return False
    
    return dfs(0, 0)

def _f_val(m):
    """
    Compute f(12, m): the LEAST positive L such that (m, m+L] contains
    distinct b_1,...,b_12 with i dividing b_i.
    """
    # Binary search for the minimum L
    lo = 1
    hi = MAX_L  # 144 is the upper bound
    
    # First check if hi works
    if not _can_match(m, hi):
        # Shouldn't happen per problem statement, but handle gracefully
        return hi
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m).
    Uses deadline guard to ensure we finish within time limits.
    """
    deadline = time.time() + 80  # 80s deadline, leaving 10s margin
    
    best_m = 12
    best_f = _f_val(12)
    
    # Try a range of m values. We'll sample strategically.
    # The key insight: f(12, m) depends on the structure of multiples in (m, m+L].
    # For small L, it's easier to find matches. For large L, the constraint is tighter.
    # We want to maximize the minimum L needed, which means we want m where
    # the interval (m, m+L] is "sparse" in terms of divisible numbers.
    
    # Strategy: search over m values, focusing on ranges where the density
    # of multiples might be lower.
    
    # Let's try a grid search with refinement
    # First pass: coarse grid
    step = 1000
    for m in range(0, MAX_M + 1, step):
        if time.time() > deadline - 5:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # Second pass: finer grid around promising regions
    if time.time() < deadline - 10:
        step2 = 100
        for m in range(0, MAX_M + 1, step2):
            if time.time() > deadline - 5:
                break
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m
    
    # Third pass: even finer
    if time.time() < deadline - 5:
        step3 = 10
        for m in range(0, MAX_M + 1, step3):
            if time.time() > deadline - 2:
                break
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m
    
    # Final pass: local refinement around best_m
    if time.time() < deadline - 1:
        for m in range(max(0, best_m - 50), min(MAX_M, best_m + 50) + 1):
            if time.time() > deadline - 0.5:
                break
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-END
