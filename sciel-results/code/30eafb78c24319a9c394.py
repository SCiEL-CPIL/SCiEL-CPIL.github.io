"""Improved program using bitmask-based local search with 2-adic valuation insights."""
import time

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    # 1. Precompute conflicts
    # conflicts[i] is a set of j such that (i+1)*(j+1) % (i+1+j+1) == 0
    # We use 0-indexed for bitmasks, so value v is index v-1
    # But 256 bits fits in a Python int easily.
    
    # Let's use a list of sets for clarity first, then convert if needed.
    # Actually, for speed, precompute a list of lists of conflicts for each number 1..256
    conflicts = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a].append(b)
                conflicts[b].append(a)
                
    # Initial Seed: All Odds + Powers of 2
    S = set()
    for i in range(1, N + 1):
        if i % 2 != 0:
            S.add(i)
    for p in [1, 2, 4, 8, 16, 32, 64, 128]:
        S.add(p)
        
    # Convert S to a set for O(1) lookup
    # We will maintain S as a set.
    
    # Helper to check if a number can be added
    def can_add(x, current_S):
        for y in current_S:
            if x in conflicts[y]: # O(|conflicts[x]|) if we check intersection? 
                # Better: check if any neighbor of x is in S
                pass
        # This is slow. Let's precompute a bitmask for S?
        # 256 bits.
        pass
        
    # Bitmask approach
    # bit i set if (i+1) is in S
    mask_S = 0
    for x in S:
        mask_S |= (1 << (x - 1))
        
    # Precompute conflict masks for each number
    # conflict_mask[i] has bit j set if (i+1) and (j+1) conflict
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        m = 0
        for b in conflicts[a]:
            m |= (1 << (b - 1))
        conflict_mask[a] = m
            
    # Check if x can be added to S represented by mask_S
    # x can be added if conflict_mask[x] & mask_S == 0
    # Note: x is not in S yet, so we don't need to check self-conflict (which is 0 anyway)
    
    def try_add(x):
        nonlocal mask_S
        if mask_S & (1 << (x - 1)):
            return False # Already in S
        if conflict_mask[x] & mask_S:
            return False
        mask_S |= (1 << (x - 1))
        return True
        
    def try_remove(x):
        nonlocal mask_S
        if not (mask_S & (1 << (x - 1))):
            return False
        mask_S &= ~(1 << (x - 1))
        return True
        
    def get_size():
        return bin(mask_S).count("1")
        
    def get_set():
        return [i + 1 for i in range(N) if (mask_S >> i) & 1]
        
    # Greedy Fill
    # Try to add any remaining number that fits
    improved = True
    while improved:
        improved = False
        for x in range(1, N + 1):
            if not (mask_S & (1 << (x - 1))):
                if try_add(x):
                    improved = True
    
    # Local Search
    # Strategies:
    # 1. 1-out 1-in
    # 2. 1-out 2-in
    # 3. 2-out 1-in
    # 4. 2-out 2-in
    
    # To speed up, we can iterate through candidates for 'in' and see if removing 'out' allows it.
    
    def local_search_1_1():
        nonlocal mask_S
        improved = False
        for out_x in range(1, N + 1):
            if not (mask_S & (1 << (out_x - 1))):
                continue
            # Temporarily remove out_x
            mask_S &= ~(1 << (out_x - 1))
            
            # Try to add any in_y
            for in_y in range(1, N + 1):
                if mask_S & (1 << (in_y - 1)):
                    continue
                if not (conflict_mask[in_y] & mask_S):
                    # Check if in_y conflicts with out_x? No, we removed out_x.
                    # So if in_y is compatible with current S (which doesn't have out_x), it can be added.
                    mask_S |= (1 << (in_y - 1))
                    improved = True
                    break
            # If we didn't add anything, restore out_x
            if not improved:
                mask_S |= (1 << (out_x - 1))
            else:
                # We added one, so S size is same. 
                # But maybe we can add more? The loop breaks after one add.
                # We should continue checking for more adds?
                # For 1-1, we just swap.
                # Restart the outer loop? Or continue?
                # Breaking and restarting is safer for convergence.
                return True
        return False

    def local_search_1_2():
        nonlocal mask_S
        improved = False
        for out_x in range(1, N + 1):
            if not (mask_S & (1 << (out_x - 1))):
                continue
            mask_S &= ~(1 << (out_x - 1))
            
            added = []
            for in_y in range(1, N + 1):
                if mask_S & (1 << (in_y - 1)):
                    continue
                if not (conflict_mask[in_y] & mask_S):
                    mask_S |= (1 << (in_y - 1))
                    added.append(in_y)
                    if len(added) >= 2:
                        break
            
            if len(added) > 1: # Added 2, removed 1. Net +1
                improved = True
                return True
            else:
                # Revert
                for a in added:
                    mask_S &= ~(1 << (a - 1))
                mask_S |= (1 << (out_x - 1))
        return False

    def local_search_2_1():
        nonlocal mask_S
        improved = False
        # Iterate pairs
        for i in range(1, N + 1):
            if not (mask_S & (1 << (i - 1))):
                continue
            for j in range(i + 1, N + 1):
                if not (mask_S & (1 << (j - 1))):
                    continue
                mask_S &= ~(1 << (i - 1))
                mask_S &= ~(1 << (j - 1))
                
                for in_y in range(1, N + 1):
                    if mask_S & (1 << (in_y - 1)):
                        continue
                    if not (conflict_mask[in_y] & mask_S):
                        mask_S |= (1 << (in_y - 1))
                        # Net 0 change? 2 out, 1 in. Size -1. 
                        # We want to maximize size. So 2-out 1-in is bad unless it enables further improvements?
                        # Usually we look for moves that increase size.
                        # So 2-out 1-in is not a primary target for size increase.
                        # Skip this strategy for now, focus on 1-out 2-in and 2-out 3-in.
                        mask_S &= ~(1 << (in_y - 1))
                        break
                mask_S |= (1 << (i - 1))
                mask_S |= (1 << (j - 1))
        return False

    def local_search_2_3():
        nonlocal mask_S
        improved = False
        for i in range(1, N + 1):
            if not (mask_S & (1 << (i - 1))):
                continue
            for j in range(i + 1, N + 1):
                if not (mask_S & (1 << (j - 1))):
                    continue
                mask_S &= ~(1 << (i - 1))
                mask_S &= ~(1 << (j - 1))
                
                added = []
                for in_y in range(1, N + 1):
                    if mask_S & (1 << (in_y - 1)):
                        continue
                    if not (conflict_mask[in_y] & mask_S):
                        mask_S |= (1 << (in_y - 1))
                        added.append(in_y)
                        if len(added) >= 3:
                            break
                
                if len(added) >= 3:
                    improved = True
                    return True
                else:
                    for a in added:
                        mask_S &= ~(1 << (a - 1))
                    mask_S |= (1 << (i - 1))
                    mask_S |= (1 << (j - 1))
        return False

    # Run loops
    while time.time() - start_time < time_limit:
        if local_search_1_2():
            continue
        if local_search_1_1():
            continue
        if local_search_2_3():
            continue
        # If no improvement in a full pass, break?
        # To avoid infinite loop, we can track if any improvement happened in the last cycle.
        # But the functions return True only if improvement.
        # If all return False, we are stuck.
        break
        
    return get_set()
# EVOLVE-BLOCK-END
