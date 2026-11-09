import numpy as np
import random
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Precompute Conflict Masks
    # conflict_mask[i] is a bitmask of all nodes j such that (i-j) % m is in forbidden
    # Since m=205, we can use Python's arbitrary precision integers as bitsets.
    # Bit i is set if node i is in the set.
    
    # We need to check if two sets are disjoint.
    # Let's precompute for each node i, a bitmask `conflict_mask[i]` 
    # where bit j is 1 if j conflicts with i.
    
    conflict_masks = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_masks[i] = mask

    def is_valid_subset(selected_mask):
        # Check if any two nodes in selected_mask conflict
        # This is O(N^2) if done naively, but we can do it smarter or just rely on construction.
        # For local search, we maintain validity incrementally.
        pass

    # Strategy: 
    # 1. Start with an empty set.
    # 2. Greedily add nodes in random order.
    # 3. Local Search: Try to add a node. If none can be added, try to remove one and add two? 
    #    Or simple swap: remove a random selected node, try to add a random unselected node.
    #    Actually, a "kick" strategy: remove a few random nodes, then greedily fill again.

    best_set = 0
    best_size = 0
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')

    # Helper to check if adding node `node` to `current_mask` is valid
    def can_add(current_mask, node):
        # If node is already in mask, return False (or handle separately)
        if (current_mask >> node) & 1:
            return False
        # Check if any neighbor of `node` is in `current_mask`
        # Neighbors of `node` are in `conflict_masks[node]`
        # Intersection with `current_mask` must be 0
        return (current_mask & conflict_masks[node]) == 0

    # Greedy fill from a starting mask
    def greedy_fill(start_mask, order):
        current_mask = start_mask
        for node in order:
            if can_add(current_mask, node):
                current_mask |= (1 << node)
        return current_mask

    # Local Search: 
    # We will run multiple iterations.
    # 1. Generate a random permutation of nodes.
    # 2. Greedy fill to get a maximal independent set.
    # 3. Try to improve by swapping.
    
    # A more powerful local search:
    # Maintain a set S.
    # While improved:
    #   Try to add a random node. If success, continue.
    #   If no node can be added, pick a random node in S, remove it, and try to add a random node not in S.
    #   If that doesn't increase size, maybe try a "2-opt" style: remove 2, add 3?
    #   Given the small size, simple swaps might suffice.

    # Let's try a specific heuristic: 
    # Since the graph is vertex-transitive (Cayley graph), we can fix one node (e.g., 0) 
    # and search for the best set containing 0? 
    # Actually, by symmetry, the optimal set might not contain 0, but we can shift the set.
    # However, the set of forbidden differences is symmetric.
    
    # Let's just run a randomized local search with many restarts.
    
    max_time = 500 # seconds
    start_time = time.time()
    
    # Precompute list of nodes
    nodes = list(range(m))
    
    while time.time() - start_time < max_time:
        # Random restart
        random.shuffle(nodes)
        current_mask = 0
        # Greedy phase
        for node in nodes:
            if can_add(current_mask, node):
                current_mask |= (1 << node)
        
        # Local Search Phase
        # Try to add any node
        improved = True
        while improved:
            improved = False
            # Try to add a random node
            random.shuffle(nodes)
            for node in nodes:
                if can_add(current_mask, node):
                    current_mask |= (1 << node)
                    improved = True
                    break # restart the loop to try adding more
            
            if not improved:
                # Try to swap
                # Get list of selected and unselected
                selected_nodes = [i for i in range(m) if (current_mask >> i) & 1]
                unselected_nodes = [i for i in range(m) if not ((current_mask >> i) & 1)]
                
                if not selected_nodes or not unselected_nodes:
                    break
                
                # Try a few random swaps
                for _ in range(100):
                    s_node = random.choice(selected_nodes)
                    u_node = random.choice(unselected_nodes)
                    
                    # Remove s_node
                    new_mask = current_mask & ~(1 << s_node)
                    
                    # Try to add u_node to new_mask
                    if can_add(new_mask, u_node):
                        # Check if this swap is beneficial? 
                        # It keeps size same. We need to see if it allows further growth.
                        # For now, accept it if it doesn't decrease size (it doesn't).
                        # To avoid cycles, we can accept with some probability or just accept and hope.
                        # Better: Only accept if it leads to a state where we can add more?
                        # That's expensive. 
                        # Let's just accept the swap to explore the space.
                        current_mask = new_mask | (1 << u_node)
                        improved = True
                        break # restart adding phase
                
                if not improved:
                    # Try a "kick": remove 2 random nodes, then greedily fill
                    if len(selected_nodes) >= 2:
                        s1, s2 = random.sample(selected_nodes, 2)
                        new_mask = current_mask & ~(1 << s1) & ~(1 << s2)
                        # Greedy fill
                        random.shuffle(nodes)
                        for node in nodes:
                            if can_add(new_mask, node):
                                new_mask |= (1 << node)
                        
                        if popcount(new_mask) > popcount(current_mask):
                            current_mask = new_mask
                            improved = True
                            break
                        elif popcount(new_mask) == popcount(current_mask):
                            # Accept with some probability to escape local optima
                            if random.random() < 0.1:
                                current_mask = new_mask
                                improved = True
                                break

        # Update best
        if popcount(current_mask) > best_size:
            best_size = popcount(current_mask)
            best_set = current_mask
            # print(f"New best: {best_size}")

    # Convert best_set mask to array
    result = []
    for i in range(m):
        if (best_set >> i) & 1:
            result.append(i)
    
    return np.array(result, dtype=np.int64)
