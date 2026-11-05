import time
import random

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin for 60s limit
    
    n = 512
    
    # Generate base set: numbers in [1, n] with base-3 digits 0 or 1
    base_set = set()
    for i in range(1, n + 1):
        # Check if base-3 representation has only 0s and 1s
        x = i
        valid = True
        while x > 0:
            if x % 3 == 2:
                valid = False
                break
            x //= 3
        if valid:
            base_set.add(i)
            
    # Helper to check if adding x to a set (represented by bool array) is valid
    def is_valid_addition(x, in_set):
        # Case 1: x is the first term (a)
        # a, a+d, a+2d, a+3d
        # x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
                return False
        
        # Case 2: x is the second term (a+d)
        # a, a+d, a+2d, a+3d
        # x-d, x, x+d, x+2d
        # d must satisfy x-d >= 1 and x+2d <= n
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d]:
                return False
        
        # Case 3: x is the third term (a+2d)
        # a, a+d, a+2d, a+3d
        # x-2d, x-d, x, x+d
        # d must satisfy x-2d >= 1 and x+d <= n
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if in_set[x - 2 * d] and in_set[x - d] and in_set[x + d]:
                return False
        
        # Case 4: x is the fourth term (a+3d)
        # a, a+d, a+2d, a+3d
        # x-3d, x-2d, x-d, x
        # d must satisfy x-3d >= 1
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if in_set[x - 3 * d] and in_set[x - 2 * d] and in_set[x - d]:
                return False
                
        return True

    best_set = set(base_set)
    best_size = len(base_set)
    
    # Candidates are all numbers not in base_set
    all_nums = set(range(1, n + 1))
    candidates = list(all_nums - base_set)
    
    num_restarts = 1000
    # Adjust restarts based on time if necessary, but 1000 should be fast enough
    
    for _ in range(num_restarts):
        if time.time() - start_time > limit:
            break
            
        current_set = set(base_set)
        in_set = [False] * (n + 1)
        for x in current_set:
            in_set[x] = True
            
        # Shuffle candidates for this restart
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(x, in_set):
                current_set.add(x)
                in_set[x] = True
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    return sorted(best_set)
