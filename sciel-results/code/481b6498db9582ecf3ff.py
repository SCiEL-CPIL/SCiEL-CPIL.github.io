import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    n = 12
    max_m = 1000000
    R = 27720  # lcm(1..12)
    
    f_cache = {}
    
    def check_matching(r, L):
        # match_to[x] = i
        match_to = [-1] * (L + 1)
        
        def dfs(i, visited):
            for x in range(1, L + 1):
                if visited[x]:
                    continue
                if (r + x) % i == 0:
                    visited[x] = True
                    if match_to[x] == -1 or dfs(match_to[x], visited):
                        match_to[x] = i
                        return True
            return False
        
        match_count = 0
        for i in range(1, n + 1):
            visited = [False] * (L + 1)
            if dfs(i, visited):
                match_count += 1
        
        return match_count == n

    def compute_f(r):
        if r in f_cache:
            return f_cache[r]
        
        lo = 12
        hi = 144
        ans = 144
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if check_matching(r, mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        f_cache[r] = ans
        return ans

    best_r = 0
    max_L_for_residue = -1
    
    for r in range(R):
        if time.time() > deadline:
            break
            
        L_val = compute_f(r)
        if L_val > max_L_for_residue:
            max_L_for_residue = L_val
            best_r = r
            
    return int(best_r)

# EVOLVE-BLOCK-END
