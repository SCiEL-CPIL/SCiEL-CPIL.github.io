import time
import random

def construct():
    N = 256
    # Precompute conflicts
    # i, j are 0-indexed for 1..256
    conflicts = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflicts[i] |= (1 << j)
                
    # Initial best: All odds
    # Odds are 1, 3, 5... which are indices 0, 2, 4...
    best_mask = 0
    for i in range(0, N, 2):
        best_mask |= (1 << i)
    best_size = bin(best_mask).count('1')
    
    start_time = time.time()
    time_limit = 85.0
    
    all_nodes = list(range(N))
    
    while time.time() - start_time < time_limit:
        # Shake: remove some random nodes from best_mask
        # Number of nodes to remove: random between 1 and 10?
        # Or proportional?
        num_remove = random.randint(1, 15)
        
        # Get list of indices in best_mask
        # We need to remove random ones.
        # It's easier to just pick random indices and clear them if they are set.
        # But we want to remove exactly num_remove set bits.
        # Let's collect set bits.
        set_bits = []
        m = best_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            set_bits.append(idx)
            m ^= lsb
            
        if len(set_bits) < num_remove:
            num_remove = len(set_bits)
            
        if num_remove == 0:
            continue
            
        # Shuffle and pick
        random.shuffle(set_bits)
        to_remove = set_bits[:num_remove]
        
        current_mask = best_mask
        for idx in to_remove:
            current_mask &= ~(1 << idx)
            
        # Now refill
        # Unselected nodes are those not in current_mask
        # We can just iterate through all nodes in random order
        # and add if valid.
        # This is O(N) per iteration.
        
        # To make it faster, we can maintain a list of unselected nodes?
        # But N=256 is small.
        
        random.shuffle(all_nodes)
        
        added_count = 0
        for u in all_nodes:
            if not (current_mask & (1 << u)):
                # u is not selected
                if (current_mask & conflicts[u]) == 0:
                    current_mask |= (1 << u)
                    added_count += 1
                    
        current_size = best_size - num_remove + added_count
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
    # Convert to list
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
