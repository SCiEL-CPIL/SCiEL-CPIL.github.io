import random
import time
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin

    # Precompute 4-APs as bitmasks
    # AP is (a, a+d, a+2d, a+3d)
    # We use 0-indexed bits: element i is bit (i-1)
    aps = []
    aps_by_element = [[] for _ in range(n)]
    
    for a in range(1, n+1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            idx = len(aps)
            aps.append(mask)
            for x in (a, b, c, e):
                aps_by_element[x-1].append(idx)
                
    num_aps = len(aps)
    
    # State
    best_size = 0
    best_S = []
    
    # We will use a randomized local search (WalkSAT style)
    # Objective: minimize number of violated 4-APs (i.e., APs fully contained in S)
    # But we also want to maximize size.
    # Actually, the constraint is "no 4-AP". So we need a valid set.
    # We can use a fitness function that penalizes violations heavily, or we can just
    # try to find a valid set with large size.
    
    # Let's try a hybrid approach:
    # 1. Generate a random subset.
    # 2. Remove elements from violating APs until no AP is violated.
    # 3. Try to add elements back if possible (greedy fill).
    # 4. Repeat with different random seeds/orders.
    
    # This is similar to the original program but potentially faster with bitmasks.
    
    # However, the "inspiration" and "directive" suggest WalkSAT.
    # Let's stick to a robust randomized greedy + local search which is known to work well for such constraints.
    # The key improvement from the "knowledge" is bitset representation for speed.
    
    def greedy_fill(state_mask):
        # Add all available elements that don't create a 4-AP
        # An element x is available if for all APs containing x, the other 3 elements are not all in S.
        # Wait, that's not right. If we add x, we create a violation if there exists an AP (x, y, z, w) 
        # such that y,z,w are already in S.
        # So x is available if for all APs containing x, not all other 3 are in S.
        
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if not (state_mask & (1 << x)):
                    # Check if x can be added
                    can_add = True
                    for ap_idx in aps_by_element[x]:
                        mask = aps[ap_idx]
                        # Check if the other 3 elements are in state_mask
                        # mask without x
                        other_mask = mask & ~(1 << x)
                        # If other_mask is subset of state_mask, then adding x creates a 4-AP
                        if (other_mask & state_mask) == other_mask:
                            can_add = False
                            break
                    if can_add:
                        state_mask |= (1 << x)
                        changed = True
        return state_mask

    def remove_violations(state_mask):
        # While there is a violated AP, remove a random element from it
        while True:
            # Find a violated AP
            violated_idx = -1
            for i, mask in enumerate(aps):
                if (state_mask & mask) == mask:
                    violated_idx = i
                    break
            
            if violated_idx == -1:
                break
            
            # Pick a random element from the violated AP to remove
            ap_mask = aps[violated_idx]
            candidates = []
            for x in range(n):
                if ap_mask & (1 << x):
                    candidates.append(x)
            
            x_to_remove = random.choice(candidates)
            state_mask &= ~(1 << x_to_remove)
        
        return state_mask

    # Main Loop
    num_restarts = 0
    while time.time() - start_time < time_limit:
        # Random initialization
        # We can start with a random subset of density p
        # Or just start empty and fill?
        # Or start with all elements and remove violations?
        # Starting with all elements and removing violations is a good way to get a large set.
        # But it might get stuck in a local minimum of small size.
        # Let's try both:
        # 1. Start with random subset, fill greedily.
        # 2. Start with all, remove violations.
        
        # Strategy 1: Random subset + greedy fill
        p = random.uniform(0.3, 0.7)
        state_mask = 0
        for i in range(n):
            if random.random() < p:
                state_mask |= (1 << i)
        
        state_mask = greedy_fill(state_mask)
        size = bin(state_mask).count('1')
        
        if size > best_size:
            best_size = size
            best_S = [i+1 for i in range(n) if state_mask & (1 << i)]
        
        # Strategy 2: All elements + remove violations
        state_mask2 = (1 << n) - 1
        state_mask2 = remove_violations(state_mask2)
        size2 = bin(state_mask2).count('1')
        
        if size2 > best_size:
            best_size = size2
            best_S = [i+1 for i in range(n) if state_mask2 & (1 << i)]
            
        num_restarts += 1
        
        # If we've done many restarts, we might be hitting diminishing returns.
        # We can increase the aggressiveness of local search or change strategy.
        # For now, just continue.
        
    return best_S

# Test with a small example to ensure it works
# construct()
