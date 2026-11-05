import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe against 150s limit
    
    n = 9
    vertices = range(n)
    all_edges = list(itertools.combinations(vertices, 5))
    num_edges = len(all_edges)
    
    # Map edge to index
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute MonoMask for all 2^n colorings
    # MonoMask[M] is a bitmask of edges that are monochromatic under coloring M
    num_masks = 1 << n
    mono_masks = [0] * num_masks
    
    # Optimization: Precompute vertex colors for each mask?
    # Or just iterate.
    for m in range(num_masks):
        # Determine color of each vertex
        colors = [(m >> v) & 1 for v in vertices]
        mask_val = 0
        for i, e in enumerate(all_edges):
            # Check if all vertices in e have the same color
            c0 = colors[e[0]]
            is_mono = True
            for v in e[1:]:
                if colors[v] != c0:
                    is_mono = False
                    break
            if is_mono:
                mask_val |= (1 << i)
        mono_masks[m] = mask_val
    
    best_edges_count = num_edges
    best_mask = (1 << num_edges) - 1  # All edges present
    
    # Function to check if a hypergraph represented by bitmask is 2-colorable
    def is_2_colorable(h_mask):
        for m in range(num_masks):
            if (h_mask & mono_masks[m]) == 0:
                return True
        return False
    
    # Greedy removal with random restarts
    # We try to remove edges one by one. If removal preserves non-2-colorability, we keep it removed.
    # We repeat this process with different random orders of edge indices.
    
    max_restarts = 50
    restart_count = 0
    
    while restart_count < max_restarts:
        if time.time() > deadline - 5.0:
            break
            
        current_mask = (1 << num_edges) - 1
        # Create a random permutation of edge indices to check
        order = list(range(num_edges))
        random.shuffle(order)
        
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in order:
                if not (current_mask & (1 << i)):
                    continue # Edge already removed
                
                # Try removing edge i
                test_mask = current_mask ^ (1 << i)
                
                if not is_2_colorable(test_mask):
                    current_mask = test_mask
                    improved = True
                    break # Restart the inner loop with the new mask and same order? 
                         # Actually, breaking and restarting the 'improved' loop 
                         # ensures we check the new state from the beginning of the order.
                         # This is a standard greedy approach.
            
        if current_mask.bit_count() < best_edges_count:
            best_edges_count = current_mask.bit_count()
            best_mask = current_mask
            
        restart_count += 1
        
    # Convert best_mask back to list of edges
    final_edges = []
    for i in range(num_edges):
        if best_mask & (1 << i):
            final_edges.append(list(all_edges[i]))
            
    return {'n': n, 'edges': final_edges}
