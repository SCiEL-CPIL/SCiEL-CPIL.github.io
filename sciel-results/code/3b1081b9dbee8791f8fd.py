import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer for overhead/return
    
    best_set = [1, 2, 5, 11]
    best_size = 4
    
    MAX_VAL = 284
    MAX_SUM = 2 * MAX_VAL
    TARGET_SIZE = 20
    
    # Initial sums
    init_S = [1, 2, 5, 11]
    sums_arr = [False] * (MAX_SUM + 1)
    for i in range(len(init_S)):
        for j in range(i, len(init_S)):
            sums_arr[init_S[i] + init_S[j]] = True
            
    # List of all candidates
    all_candidates = [x for x in range(1, MAX_VAL + 1) if x not in init_S]
    
    # Random seed for reproducibility in dev, but random in eval?
    # random module uses system time by default.
    
    while time.time() < deadline:
        S = init_S[:]
        current_sums = sums_arr[:]
        avail = all_candidates[:]
        
        while avail:
            # Optimization: If we have already found 20, we could break outer loop?
            # No, we return immediately inside.
            
            valid_candidates = []
            # Iterate through available candidates
            for x in avail:
                # Check if x can be added
                # 1. 2x not in sums
                # 2. x+y not in sums for all y in S
                if current_sums[2 * x]:
                    continue
                    
                conflict = False
                for y in S:
                    if current_sums[x + y]:
                        conflict = True
                        break
                        
                if not conflict:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Pick a candidate
            # To increase diversity, pick randomly
            x = random.choice(valid_candidates)
            
            # Update state
            S.append(x)
            current_sums[2 * x] = True
            # Update sums with new x
            for y in S[:-1]:
                current_sums[x + y] = True
                
            # Remove x from avail
            # Using list removal is O(N). 
            # Since N is small (280), it's fine.
            avail.remove(x)
            
            # Check if we reached target
            if len(S) >= TARGET_SIZE:
                return S
                
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
                
    return best_set
