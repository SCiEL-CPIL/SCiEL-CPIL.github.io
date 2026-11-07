import time

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    """
    # Get the numbers in the interval
    nums = list(range(m + 1, m + L + 1))
    
    # For each i in 1..n, find indices in nums that are divisible by i
    options = []
    for i in range(1, n + 1):
        opts = []
        for idx, v in enumerate(nums):
            if v % i == 0:
                opts.append(idx)
        if not opts:
            return False
        options.append((len(opts), i, opts))
        
    # Sort by number of options (most constrained first)
    options.sort(key=lambda x: x[0])
    
    used = [False] * L
    
    def dfs(idx):
        if idx == n:
            return True
        _, i, opts = options[idx]
        for pos in opts:
            if not used[pos]:
                used[pos] = True
                if dfs(idx + 1):
                    return True
                used[pos] = False
        return False
        
    return dfs(0)

def _f_val(m, n=12):
    """
    Find the least positive L such that (m, m+L] contains a valid matching.
    """
    # Max L is stated to be at most 144
    for L in range(1, 145):
        if _can_match(m, L, n):
            return L
    return 144 # Fallback, shouldn't happen

def construct():
    """
    Find m in 0..1000000 that maximizes f(12, m).
    """
    # The function f(12, m) is periodic with period lcm(1..12) = 27720
    # So we only need to search m in [0, 27720)
    K = 27720
    best_m = 0
    best_L = 0
    
    # Deadline guard, though 27720 iterations is fast
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds margin
    
    for m in range(K):
        if time.time() > deadline:
            break
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
            # If we hit the theoretical max, we can stop early
            if best_L == 144:
                break
                
    return int(best_m)
