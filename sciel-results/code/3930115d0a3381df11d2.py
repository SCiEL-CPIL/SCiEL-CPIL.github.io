import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    best_m = 12
    best_L = 0
    
    # Max L is 144
    MAX_L = 144
    MIN_L = 12
    PERIOD = 27720
    
    def get_mask(m, i, L):
        rem = (-m) % i
        mask = 0
        if rem == 0:
            start = i
        else:
            start = rem
        # j ranges from start to L step i
        # bit position is j-1
        for j in range(start, L + 1, i):
            mask |= (1 << (j - 1))
        return mask

    def has_matching(m, L):
        masks = [0] * 13
        for i in range(1, 13):
            masks[i] = get_mask(m, i, L)
        
        # Order indices by popcount (most constrained first)
        indices = list(range(1, 13))
        indices.sort(key=lambda i: masks[i].bit_count())
        
        # Memoization for DFS? 
        # State: (k, used_mask)
        # k is 0..11
        # used_mask is up to 2^144, but we only visit few states
        # Let's use a dictionary for memoization per call
        memo = {}
        
        def dfs(k, used_mask):
            if k == 12:
                return True
            key = (k, used_mask)
            if key in memo:
                return memo[key]
            
            i = indices[k]
            avail = masks[i] & ~used_mask
            while avail:
                lsb = avail & -avail
                # j = lsb.bit_length() - 1
                if dfs(k + 1, used_mask | lsb):
                    memo[key] = True
                    return True
                avail ^= lsb
            
            memo[key] = False
            return False
        
        return dfs(0, 0)

    for m in range(PERIOD):
        if time.time() > deadline:
            break
        
        low = MIN_L
        high = MAX_L
        
        # Binary search for the first L where matching exists
        while low < high:
            mid = (low + high) // 2
            if has_matching(m, mid):
                high = mid
            else:
                low = mid + 1
        
        # low is the minimal L
        if low > best_L:
            best_L = low
            best_m = m

    return int(best_m)
