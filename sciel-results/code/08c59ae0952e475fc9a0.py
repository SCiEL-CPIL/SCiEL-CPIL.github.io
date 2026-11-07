"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses bitmask DFS with most-constrained-first ordering.
    """
    if L < n:
        return False
    options = []
    for i in range(1, n + 1):
        mask = 0
        target = (-(m + 1)) % i
        # j in 0..L-1 where (m+1+j) % i == 0
        # j % i == target
        first = target
        if first >= L:
            return False
        for j in range(first, L, i):
            mask |= (1 << j)
        if mask == 0:
            return False
        options.append((mask.bit_count(), mask))
    
    options.sort(key=lambda x: x[0])
    
    sys.setrecursionlimit(10000)
    
    def dfs(k, used_mask):
        if k == n:
            return True
        cnt, mask = options[k]
        avail = mask & ~used_mask
        while avail:
            lb = avail & (-avail)
            j = lb.bit_length() - 1
            if dfs(k + 1, used_mask | lb):
                return True
            avail ^= lb
        return False
    
    return dfs(0, 0)

def _f_val(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Lower bound: at least n
    lo = n
    # Upper bound: by pigeonhole, L = n * max(i) should suffice? Actually LCM-based bound.
    # Safe upper bound: n * 12 = 144 (since each i needs at least one multiple, and spacing is at most i)
    # Actually worst case: for i=12, multiples are 12 apart, so we might need up to 12*12=144
    hi = n * n  # 144
    
    # Binary search for minimum L
    ans = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    n = 12
    # LCM of 1..12
    import math
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
    # lcm_val = 27720
    
    # Precompute f(12, m) for m = 0, 1, ..., lcm_val-1
    # f(12, m) depends only on m mod lcm_val
    best_m = 0
    best_f = 0
    
    # First pass: evaluate all residues
    f_values = [0] * lcm_val
    for m in range(lcm_val):
        if time.time() > deadline - 5:
            break
        f_values[m] = _f_val(m, n)
        if f_values[m] > best_f:
            best_f = f_values[m]
            best_m = m
    
    # Find the m in [0, 1000000] with the best f value
    # f(12, m) = f(12, m % lcm_val)
    # So we want m in [0, 1000000] such that m % lcm_val = best_m
    # The smallest such m is best_m (if best_m <= 1000000)
    # But we might be able to do better by checking if there's a residue with higher f
    
    # Actually we already found best_m as the residue with max f.
    # The actual m to return is the smallest m in [0, 1000000] with m % lcm_val == best_m
    # That's just best_m if best_m <= 1000000, else best_m + k*lcm_val for some k
    
    result_m = best_m
    if result_m > 1000000:
        # Should not happen since lcm_val = 27720 < 1000000
        result_m = best_m % (1000001)
    
    return int(result_m)

# EVOLVE-BLOCK-END
