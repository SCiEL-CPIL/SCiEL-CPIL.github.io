import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 512
    
    # Precompute conflict masks for each element x.
    # conflict_masks[x] is a list of masks. Each mask is an integer with bits set
    # at positions corresponding to the other 3 elements in a 4-AP containing x.
    # Bit i corresponds to element i+1.
    conflict_masks = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Convert to 0-based indices for bit manipulation
            i_a = a - 1
            i_b = b - 1
            i_c = c - 1
            i_e = e - 1
            
            # For element a, the others are b, c, e
            mask_a = (1 << i_b) | (1 << i_c) | (1 << i_e)
            conflict_masks[a].append(mask_a)
            
            # For element b, the others are a, c, e
            mask_b = (1 << i_a) | (1 << i_c) | (1 << i_e)
            conflict_masks[b].append(mask_b)
            
            # For element c, the others are a, b, e
            mask_c = (1 << i_a) | (1 << i_b) | (1 << i_e)
            conflict_masks[c].append(mask_c)
            
            # For element e, the others are a, b, c
            mask_e = (1 << i_a) | (1 << i_b) | (1 << i_c)
            conflict_masks[e].append(mask_e)
            
    def can_add(x, S_mask):
        """
        Check if x can be added to the set represented by S_mask.
        S_mask is an integer where bit i is 1 if element i+1 is in the set.
        """
        for mask in conflict_masks[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    def get_salem_spencer_mask():
        """
        Generate the Salem-Spencer set: numbers in [1, n] whose base-3
        representation contains only digits 0 and 1.
        """
        mask = 0
        # Powers of 3: 1, 3, 9, 27, 81, 243
        # Max sum is 1+3+9+27+81+243 = 364, which is < 512.
        # So all 2^6 = 64 combinations are valid and within range.
        for i in range(6):
            for j in range(1 << i):
                # This logic is getting complicated. Just iterate all subsets of powers of 3.
                pass
        
        # Simpler: iterate all 64 subsets of {1, 3, 9, 27, 81, 243}
        powers = [1, 3, 9, 27, 81, 243]
        for subset in range(1, 1 << 6): # Exclude 0
            val = 0
            for k in range(6):
                if subset & (1 << k):
                    val += powers[k]
            if val <= n:
                mask |= (1 << (val - 1))
        return mask

    best_mask = 0
    best_size = 0
    
    # List of initial masks to try
    initial_masks = []
    
    # 1. Salem-Spencer baseline
    initial_masks.append(get_salem_spencer_mask())
    
    # 2. Random Greedy starts
    num_starts = 10
    for _ in range(num_starts):
        S_mask = 0
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        for x in elements:
            if can_add(x, S_mask):
                S_mask |= (1 << (x - 1))
        initial_masks.append(S_mask)
    
    # Local Search on each initial mask
    for init_mask in initial_masks:
        if time.time() > deadline:
            break
            
        S_mask = init_mask
        current_size = bin(S_mask).count('1')
        
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            
        # Local Search Loop
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any element
            # We iterate through all elements not in S_mask.
            # To be efficient, we can just loop 1..n and check if bit is not set.
            for x in range(1, n + 1):
                if not (S_mask & (1 << (x - 1))):
                    if can_add(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                        # Restart addition phase as adding one might enable others
                        break
            
            if improved:
                continue
            
            # Phase 2: No additions possible. Try swaps.
            # Pick a random element y to remove and x to add.
            # We need a list of elements currently in S for efficient random choice.
            # Since n=512, building the list is fast.
            s_list = [i + 1 for i in range(n) if S_mask & (1 << i)]
            
            if not s_list:
                break
                
            # Try a few swaps before giving up on this specific local optimum
            # and moving to the next initial mask.
            swap_attempts = 0
            max_swaps = 200
            
            while swap_attempts < max_swaps and time.time() < deadline:
                swap_attempts += 1
                
                y = random.choice(s_list)
                x = random.randint(1, n)
                
                if x == y or (S_mask & (1 << (x - 1))):
                    continue
                
                # Try swapping y with x
                # New mask = S_mask ^ (1 << (y-1))
                S_mask_without_y = S_mask & ~(1 << (y - 1))
                
                if can_add(x, S_mask_without_y):
                    # Accept swap
                    S_mask = S_mask_without_y | (1 << (x - 1))
                    # Update s_list? We can rebuild it or just mark that it's dirty.
                    # Rebuilding is safer and fast enough.
                    s_list = [i + 1 for i in range(n) if S_mask & (1 << i)]
                    
                    # Check if this swap allows an immediate addition?
                    # We can just let the outer loop handle it by breaking swap loop
                    # and going back to Phase 1.
                    break
            
            # If we exhausted swap attempts without breaking (i.e., no valid swap found 
            # or we just finished a valid swap and broke), we continue to the next 
            # iteration of the main while loop. 
            # If we broke due to a valid swap, Phase 1 will run again.
            # If we exhausted swaps, Phase 1 will run, find no additions, 
            # and we'll try swaps again. This might cycle.
            # To avoid infinite cycling in a dead-end local optimum, 
            # we could add a restart mechanism, but multi-start covers this.
            # If we are stuck, we just keep trying swaps.
            
    # Convert best_mask to sorted list
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    if not result:
        return [1, 2]
        
    return result
