import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add((x * x) % m)
    forbidden.discard(0)
    
    # 2. Build conflict graph using bitmasks
    # conflicts_mask[i] is an integer where bit j is 1 if i and j conflict
    conflicts_mask = [0] * m
    degrees = [0] * m
    
    for i in range(m):
        for j in range(i + 1, m):
            d = (i - j) % m
            if d in forbidden:
                conflicts_mask[i] |= (1 << j)
                conflicts_mask[j] |= (1 << i)
                degrees[i] += 1
                degrees[j] += 1

    # Helper to check if a node can be added to a set represented by a bitmask
    def can_add_mask(node, current_mask):
        return (conflicts_mask[node] & current_mask) == 0

    # Greedy construction with bitmask
    def greedy_construct(order):
        current_mask = 0
        for node in order:
            if can_add_mask(node, current_mask):
                current_mask |= (1 << node)
        return current_mask

    # Convert mask to list of nodes
    def mask_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    # Initial best solution
    best_mask = 0
    best_size = 0
    
    # Try a few random orders to find a decent starting point
    # Also try sorted by degree
    initial_orders = []
    for _ in range(20):
        order = list(range(m))
        random.shuffle(order)
        initial_orders.append(order)
    
    # Add degree-sorted order (ascending)
    initial_orders.append(sorted(range(m), key=lambda x: degrees[x]))
    # Add degree-sorted order (descending)
    initial_orders.append(sorted(range(m), key=lambda x: -degrees[x]))
    
    for order in initial_orders:
        mask = greedy_construct(order)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask

    start_time = time.time()
    deadline = 580.0  # Safety margin
    
    # Local Search: Kick and Polish
    while time.time() - start_time < deadline:
        # Create a working copy (mask is immutable int, so just copy value)
        current_mask = best_mask
        
        # Kick: remove k random elements
        # Determine current set size to pick k
        current_size = bin(current_mask).count('1')
        
        # Vary kick size: sometimes small, sometimes large
        if random.random() < 0.3:
            k = random.randint(1, 3)
        else:
            k = random.randint(3, 10)
            
        if current_size > k:
            # Get list of nodes in current mask
            current_nodes = mask_to_list(current_mask)
            to_remove = random.sample(current_nodes, k)
            for node in to_remove:
                current_mask &= ~(1 << node)
        
        # Polish: add nodes back
        # Candidates are nodes not in current
        # We want to add as many as possible.
        # Strategy: Iterate through candidates.
        # To make it deterministic-ish but varied, we can shuffle candidates or sort by degree.
        
        candidates = [i for i in range(m) if not (current_mask >> i) & 1]
        
        # Choose strategy:
        # 1. Random shuffle
        # 2. Sort by degree (ascending - greedy choice)
        # 3. Sort by degree (descending)
        
        strategy = random.random()
        if strategy < 0.5:
            random.shuffle(candidates)
        elif strategy < 0.8:
            candidates.sort(key=lambda x: degrees[x])
        else:
            candidates.sort(key=lambda x: -degrees[x])
            
        for node in candidates:
            if can_add_mask(node, current_mask):
                current_mask |= (1 << node)
                
        new_size = bin(current_mask).count('1')
        
        if new_size > best_size:
            best_size = new_size
            best_mask = current_mask

    return np.array(mask_to_list(best_mask), dtype=np.int64)
