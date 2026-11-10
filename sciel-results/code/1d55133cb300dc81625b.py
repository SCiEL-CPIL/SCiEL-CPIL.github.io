import time

# EVOLVE-BLOCK-START

def _has_perfect_matching(m, L):
    n = 12
    match_to = [None] * (L + 1)
    
    def dfs(i, seen):
        # Iterate multiples of i in (m, m+L]
        # start = smallest multiple of i > m
        # end = largest multiple of i <= m+L
        start = (m // i + 1) * i
        end = (m + L) // i * i
        
        for x in range(start, end + 1, i):
            idx = x - m # 1-based index in the window
            if idx in seen:
                continue
            seen.add(idx)
            if match_to[idx] is None or dfs(match_to[idx], seen):
                match_to[idx] = i
                return True
        return False

    for i in range(1, n + 1):
        seen = set()
        if not dfs(i, seen):
            return False
    return True

def _compute_f(m):
    # Binary search for the smallest L >= 12 such that matching exists
    lo = 12
    hi = 144
    best = 144
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best

def construct():
    best_m = 0
    best_f = 0
    start_time = time.time()
    
    # Search over one period 27720
    for m in range(27720):
        if time.time() - start_time > 85:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return best_m

# EVOLVE-BLOCK-END
