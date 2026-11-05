import time

def construct():
    start = time.time()
    deadline = start + 55
    
    target = 479001600  # 12!
    
    best = [33, 28, 25, 32, 27, 24]
    best_max = 33
    
    # For max <= 32: factor 11 must be in 22 (only option since 11*3=33>32)
    # Remaining product: 12! / 22 = 21772800 = 2^9 * 3^5 * 5^2 * 7
    # Valid candidates: divisors of 21772800, in range (12, 32], not 22
    rem_target = 21772800
    cands = [14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32]
    
    def dfs(idx, rem, chosen):
        if time.time() > deadline - 2:
            return None
        if rem == 1:
            return chosen[:]
        if idx >= len(cands):
            return None
        if len(chosen) >= 11:  # max 11 more factors (total 12 with 22)
            return None
        
        d = cands[idx]
        # Try including d
        if rem % d == 0:
            result = dfs(idx + 1, rem // d, chosen + [d])
            if result is not None:
                return result
        # Try excluding d
        return dfs(idx + 1, rem, chosen)
    
    result = dfs(0, rem_target, [])
    if result is not None:
        candidate = [22] + result
        # Verify
        if (len(candidate) <= 12 and 
            len(set(candidate)) == len(candidate) and 
            all(x > 12 for x in candidate) and
            eval('*'.join(map(str, candidate))) == target):
            if max(candidate) < best_max:
                best = candidate
                best_max = max(candidate)
    
    return best

