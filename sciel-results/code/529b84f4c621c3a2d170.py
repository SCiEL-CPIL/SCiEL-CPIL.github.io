import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    P = 27720  # LCM(1..12)
    n = 12
    
    best_L = 0
    best_m = 0
    
    # Precompute modular constraints
    # For each i in 1..12, and for each possible residue r = m % i,
    # we can precompute the pattern of valid offsets d mod i.
    # But since L is small, we can just compute on the fly.
    
    def has_matching(m, L):
        # Right nodes: offsets d in 1..L
        # Left nodes: i in 1..12
        # Edge (i, d) if (m + d) % i == 0
        
        # For each i, find the list of valid d's
        valid_d = [None] * (n + 1)  # 1-indexed for i
        for i in range(1, n + 1):
            valid_d[i] = []
            # d in 1..L such that (m+d) % i == 0
            # m % i = r, so d % i = (-r) % i
            target = (-m) % i
            if target == 0:
                # d must be multiple of i
                d = i
                while d <= L:
                    valid_d[i].append(d)
                    d += i
            else:
                d = target
                while d <= L:
                    valid_d[i].append(d)
                    d += i
        
        # DFS with bitmask
        # State: mask of used d's (as an integer bitmask)
        # We assign i from 1 to n
        # To improve efficiency, order i by increasing number of valid d's (most constrained first)
        order = sorted(range(1, n + 1), key=lambda i: len(valid_d[i]))
        
        def dfs(i_idx, mask):
            if i_idx == n:
                return True
            i = order[i_idx]
            for d in valid_d[i]:
                bit = 1 << d
                if mask & bit == 0:
                    if dfs(i_idx + 1, mask | bit):
                        return True
            return False
        
        return dfs(0, 0)
    
    for m in range(P):
        if time.time() > deadline:
            break
            
        # Find minimal L for this m
        # L is at least 1, at most 144 (known bound)
        found = False
        for L in range(1, 145):
            if has_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                found = True
                break
        # If no L found up to 144, it means f(12,m) > 144, but the problem states max is 144.
        # If not found, we ignore (should not happen)
    
    # Ensure best_m is within 0..1000000
    # Since P=27720, best_m is in [0, 27719], which is within [0, 1000000]
    # We can return best_m directly.
    
    return best_m
