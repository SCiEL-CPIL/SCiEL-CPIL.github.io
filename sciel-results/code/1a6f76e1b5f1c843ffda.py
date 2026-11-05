def construct():
    import time
    import random
    import math
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    
    # Precompute all 4-APs
    all_aps = []
    for d in range(1, 171):
        for a in range(1, 513 - 3*d):
            all_aps.append((a, a+d, a+2*d, a+3*d))
            
    # Precompute AP indices for each x
    ap_indices_for_x = [[] for _ in range(N+1)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_for_x[x].append(idx)
            
    def is_valid_addition(S_mask, x):
        """Check if adding x creates any new 4-AP."""
        for idx in ap_indices_for_x[x]:
            a, b, c, d = all_aps[idx]
            others = [y for y in (a, b, c, d) if y != x]
            if all(((S_mask >> (y-1)) & 1) for y in others):
                return False
        return True

    # Greedy Initialization
    def greedy_construct(seed):
        random.seed(seed)
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        S_mask = 0
        size = 0
        for x in candidates:
            if is_valid_addition(S_mask, x):
                S_mask |= (1 << (x-1))
                size += 1
        return S_mask, size

    best_mask = 0
    best_size = 0
    
    # Phase 1: Greedy
    for i in range(50):
        if time.time() > deadline: break
        mask, size = greedy_construct(i)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Phase 2: Simulated Annealing
    current_mask = best_mask
    current_size = best_size
    
    # Use a boolean array for fast membership check in random selection logic if needed, 
    # but we only need mask for validity.
    # For random selection, we just pick random x in 1..N and check mask.
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    
    # We track "steps" to control cooling rate? No, just time.
    
    while time.time() < deadline - 2.0:
        if T < min_T:
            T = 1.0 # Reheat
            
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to Add
            x = random.randint(1, N)
            if not ((current_mask >> (x-1)) & 1): # If x is not in set
                if is_valid_addition(current_mask, x):
                    # Accept
                    current_mask |= (1 << (x-1))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    T *= cooling
                else:
                    T *= cooling
            else:
                T *= cooling
        else:
            # Try to Swap (Size 0 move)
            # Pick y in set, x not in set
            # To pick y in set efficiently, we might need a list.
            # But N=512 is small. We can just pick random y, check if in set.
            # If not, retry? Or pick another random?
            # Expected trials to pick from set is N/size ~ 2.
            
            y = random.randint(1, N)
            if (current_mask >> (y-1)) & 1: # y is in set
                x = random.randint(1, N)
                if not ((current_mask >> (x-1)) & 1): # x is not in set
                    # Tentatively remove y
                    mask_without_y = current_mask & ~(1 << (y-1))
                    if is_valid_addition(mask_without_y, x):
                        # Accept swap
                        current_mask = mask_without_y | (1 << (x-1))
                        # Size unchanged
                        T *= cooling
                    else:
                        T *= cooling
                else:
                    T *= cooling
            else:
                T *= cooling

    # Final check: ensure best_mask is valid and return sorted list
    result = [i+1 for i in range(N) if (best_mask >> i) & 1]
    return sorted(result)
