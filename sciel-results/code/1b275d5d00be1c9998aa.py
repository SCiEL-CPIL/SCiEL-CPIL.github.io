import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _compute_lcm_range(n):
    result = 1
    for i in range(1, n + 1):
        result = _lcm(result, i)
    return result

def _has_perfect_matching(m, L, n=12):
    """Check if there's a perfect matching in interval (m, m+L] for indices 1..n."""
    # For each index i, find which positions j in 1..L have (m+j) % i == 0
    # Build adjacency: adj[i] = bitmask of valid positions
    adj = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                mask |= (1 << (j - 1))
        adj[i] = mask
        if mask == 0:
            return False
    
    # Use DFS-based bipartite matching (Kuhn's algorithm)
    # match_pos[j] = which index i is matched to position j (0-indexed: j-1)
    match_pos = [-1] * L  # match_pos[j-1] = i or -1
    
    def dfs(i, visited):
        candidates = adj[i]
        while candidates:
            low_bit = candidates & (-candidates)
            j_idx = low_bit.bit_length() - 1  # 0-indexed position
            candidates ^= low_bit
            if visited & low_bit:
                continue
            visited |= low_bit
            if match_pos[j_idx] == -1 or dfs(match_pos[j_idx], visited):
                match_pos[j_idx] = i
                return True
        return False
    
    for i in range(1, n + 1):
        if not dfs(i, 0):
            return False
    return True

def _compute_f(m, n=12, max_L=200):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_1..b_n with i|b_i."""
    for L in range(1, max_L + 1):
        if _has_perfect_matching(m, L, n):
            return L
    return max_L  # Should not reach here for valid inputs

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s safety margin
    
    n = 12
    period = _compute_lcm_range(n)  # 27720
    
    best_m = 12
    best_f = _compute_f(best_m, n)
    
    # Search over m in one period
    # We can sample strategically or do full search
    # Full search over 27720 values, each requiring up to 144 matching checks
    # Each matching check is O(n * L) in worst case with DFS
    # This should be feasible within time budget
    
    m = 0
    while m < period:
        if time.time() > deadline:
            break
        
        f_val = _compute_f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            # Early exit if we hit the theoretical max
            if best_f >= 144:
                break
        
        m += 1
    
    return int(best_m)
