import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Filter self-conflicts
    # Node v is valid if 2*v is not a square
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    node_set = set(valid_nodes)
    
    # Build adjacency masks
    # Map node value to index? 
    # Let's just use the node value as the bit position directly?
    # Node 1 -> bit 0? No, node 1 -> bit 1?
    # Let's use 1-based indexing for bits to make it intuitive?
    # Bit i corresponds to node i.
    # So mask for node v is (1 << v).
    # Max bit is 256. Python ints handle this fine.
    
    adj_mask = [0] * (N + 1) # 1-indexed, adj_mask[v]
    
    for v in valid_nodes:
        for u in valid_nodes:
            if u > v and (v + u) in squares:
                adj_mask[v] |= (1 << u)
                adj_mask[u] |= (1 << v)
                
    initial_mask = 0
    for v in valid_nodes:
        initial_mask |= (1 << v)
        
    best_set = set()
    best_size = 0
    
    # Mod 3 baseline
    mod3_set = set(v for v in range(1, N+1) if v % 3 == 1)
    # Verify mod3
    # a,b in mod3 => a+b = 2 mod 3. Squares mod 3 are 0, 1. 2 is not a square.
    # Diagonal: 2a = 2 mod 3. Not a square.
    # So mod3 is valid.
    best_set = mod3_set
    best_size = len(mod3_set)
    
    # Bitmask Greedy
    # We will run multiple restarts.
    # Each restart:
    #   current_mask = initial_mask
    #   sol = []
    #   while current_mask:
    #       find min degree node v in current_mask
    #       add v to sol
    #       current_mask &= ~(adj_mask[v] | (1 << v))
        
    num_restarts = 200 # Start with a reasonable number, adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        current_mask = initial_mask
        sol = []
        
        while current_mask:
            min_deg = N + 1
            candidates = []
            
            # Iterate through set bits in current_mask
            # To do this efficiently in Python, we can loop through valid_nodes 
            # and check if bit is set.
            # Since N=256, this loop is small.
            
            for v in valid_nodes:
                if current_mask & (1 << v):
                    # Count neighbors in current_mask
                    # deg = (adj_mask[v] & current_mask).bit_count()
                    deg = (adj_mask[v] & current_mask).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
            
            if not candidates:
                break # Should not happen if current_mask is not 0
                
            # Randomly pick from candidates
            v = random.choice(candidates)
            sol.append(v)
            
            # Remove v and its neighbors
            current_mask &= ~(adj_mask[v] | (1 << v))
            
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = set(sol)
            
    return list(best_set)
