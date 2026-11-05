import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55  # seconds, keep a margin
    
    # 1. Generate the core set: numbers with base-3 digits 0 or 1
    core = set()
    for mask in range(1, 1 << 6):  # 6 digits: 3^0 to 3^5
        val = 0
        for i in range(6):
            if (mask >> i) & 1:
                val += (3 ** i)
        if 1 <= val <= n:
            core.add(val)
    
    best_set = set(core)
    best_size = len(best_set)
    
    # Helper to check if adding x to S creates a 4-AP
    # S is a set of ints.
    def can_add(S, x):
        # x is not in S
        # Check all d such that terms are in [1, n]
        # x can be a, a+d, a+2d, a+3d
        # 1. x = a: a, a+d, a+2d, a+3d. d >= 1. a+3d <= n => 3d <= n-x => d <= (n-x)//3
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        
        # 2. x = a+d: a, x, x+d, x+2d. a = x-d >= 1 => d <= x-1. x+2d <= n => 2d <= n-x => d <= (n-x)//2
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        
        # 3. x = a+2d: a, a+d, x, x+d. a = x-2d >= 1 => 2d <= x-1 => d <= (x-1)//2. x+d <= n => d <= n-x
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        
        # 4. x = a+3d: a, a+d, a+2d, x. a = x-3d >= 1 => 3d <= x-1 => d <= (x-1)//3.
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        
        return True

    candidates = [i for i in range(1, n + 1) if i not in core]
    
    # Greedy repair with random restarts/shuffles
    # We will try to add candidates to the core
    # Since the core is fixed and good, we just add others.
    
    while time.time() - start_time < time_limit:
        current_set = set(core)
        random.shuffle(candidates)
        
        for x in candidates:
            if x in current_set:
                continue
            if can_add(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            # If we found a perfect set (unlikely), we could break, but let's keep searching
            # Actually, if best_size is very high, we might want to do local search (swap)
            
    return sorted(best_set)
