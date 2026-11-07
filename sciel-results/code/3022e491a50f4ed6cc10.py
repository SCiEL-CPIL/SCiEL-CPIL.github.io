import time
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    # Bitmask representation
    # bit i represents element i+1
    mask = 0
    S = set()
    
    # Helper to check if x can be added
    def is_valid_addition(x, m, s):
        if m & (1 << (x - 1)):
            return False
        # Iterate pairs in s
        # s is a set of ints
        s_list = list(s)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                # Check if a+b is in s
                if m & (1 << (a + b - 1)):
                    # Check if x+a and x+b are in s
                    if (m & (1 << (x + a - 1))) and (m & (1 << (x + b - 1))):
                        return False
        return True

    best_size = 0
    best_mask = 0
    best_S = set()
    
    # Heuristic start: Top half
    for x in range(64, N + 1):
        mask |= (1 << (x - 1))
        S.add(x)
    
    # Greedy add from bottom
    for x in range(63, 0, -1):
        if is_valid_addition(x, mask, S):
            mask |= (1 << (x - 1))
            S.add(x)
            
    if len(S) > best_size:
        best_size = len(S)
        best_mask = mask
        best_S = S.copy()
        
    # Random Restart / Local Search
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Start with best found or random
        if rng.random() < 0.5:
            mask = best_mask
            S = best_S.copy()
        else:
            mask = 0
            S = set()
            # Random dense start
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if is_valid_addition(x, mask, S):
                    mask |= (1 << (x - 1))
                    S.add(x)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any missing element
            missing = [x for x in range(1, N + 1) if not (mask & (1 << (x - 1)))]
            rng.shuffle(missing)
            for x in missing:
                if is_valid_addition(x, mask, S):
                    mask |= (1 << (x - 1))
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_mask = mask
                        best_S = S.copy()
                    break # Restart loop to re-evaluate state
            
            if not improved:
                # Try swap: remove 1, add 2
                if len(S) > 0:
                    s_list = list(S)
                    rng.shuffle(s_list)
                    for rem in s_list:
                        if time.time() > deadline - 5:
                            break
                        # Temporarily remove
                        mask_temp = mask & ~(1 << (rem - 1))
                        S_temp = S - {rem}
                        
                        added = 0
                        missing_temp = [x for x in range(1, N + 1) if not (mask_temp & (1 << (x - 1)))]
                        rng.shuffle(missing_temp)
                        
                        for x in missing_temp:
                            if added >= 2:
                                break
                            if is_valid_addition(x, mask_temp, S_temp):
                                mask_temp |= (1 << (x - 1))
                                S_temp.add(x)
                                added += 1
                                
                        if added >= 2:
                            mask = mask_temp
                            S = S_temp
                            improved = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_mask = mask
                                best_S = S.copy()
                            break
                            
    # Final result
    result = sorted(list(best_S))
    return [int(x) for x in result]
