import random
import time

def construct():
    n = 512
    best_set = set()
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Precompute possible d ranges?
    # For x, d can be such that x-3d >= 1 and x+3d <= n?
    # Actually, we just need to check if the points are in [1, n].
    
    def is_valid_add(x, S):
        # Check if adding x creates a 4-AP
        # x can be 1st, 2nd, 3rd, 4th term
        # 1st: x, x+d, x+2d, x+3d. Need x+d, x+2d, x+3d in S.
        # Max d for 1st: x+3d <= n => d <= (n-x)//3
        for d in range(1, (n - x) // 3 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        
        # 2nd: x-d, x, x+d, x+2d. Need x-d, x+d, x+2d in S.
        # Min d: x-d >= 1 => d <= x-1. Max d: x+2d <= n => d <= (n-x)//2
        max_d_2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d_2 + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        
        # 3rd: x-2d, x-d, x, x+d. Need x-2d, x-d, x+d in S.
        # Min d: x-2d >= 1 => d <= (x-1)//2. Max d: x+d <= n => d <= n-x
        max_d_3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d_3 + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        
        # 4th: x-3d, x-2d, x-d, x. Need x-3d, x-2d, x-d in S.
        # Min d: x-3d >= 1 => d <= (x-1)//3. Max d: not restricted by upper bound of x, just d>=1
        max_d_4 = (x - 1) // 3
        for d in range(1, max_d_4 + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
                
        return True

    # We can also try to be smarter about ordering.
    # Randomized greedy
    for seed in range(10000):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Optimization: Instead of iterating through all candidates and checking validity,
        # we can just iterate. But checking validity is O(n).
        # If we have 512 candidates, and we do 512 checks, that's 512*512 ~ 260k ops per run.
        # This is fast.
        
        current_size = 0
        for x in candidates:
            if is_valid_add(x, S):
                S.add(x)
                current_size += 1
        
        if len(S) > len(best_set):
            best_set = S
            # If we reach a very large set, we might break early?
            # Known bounds for 4-AP free sets in [1, 512]:
            # r_4(512) is likely around 100-150?
            # Let's see what the seed gives.
            # Seed greedy likely gives ~100.
            # If we get > 150, it's good.
            
    return sorted(best_set)
