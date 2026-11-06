import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    # Precompute edge bitmasks for the checker
    # edge_mask[i] corresponds to all_edges[i]
    edge_masks = [0] * len(all_edges)
    for i, e in enumerate(all_edges):
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks[i] = m

    def is_non_2_colorable(edge_indices):
        """
        Returns True if the hypergraph defined by edge_indices (indices into all_edges) 
        is NOT 2-colorable.
        Uses brute-force bitmask checking.
        """
        if not edge_indices:
            return False
            
        # Create a set of masks for fast lookup? No, just iterate.
        # For n=9, 2^9 = 512 masks.
        # For each mask, check all edges in edge_indices.
        
        # Optimization: Precompute the masks for the current set?
        # Since we are removing edges one by one, we can just pass the list of masks.
        
        # Let's pass the list of masks directly to avoid index mapping overhead in the inner loop.
        # But the caller has indices. Let's just use indices and look up masks.
        
        for mask in range(1 << n):
            is_mono_found = False
            for idx in edge_indices:
                em = edge_masks[idx]
                # Check if all bits in em are 0 or all are 1 in mask
                if (mask & em) == 0 or (mask & em) == em:
                    is_mono_found = True
                    break
            if not is_mono_found:
                # Found a valid 2-coloring
                return False
                
        # No valid 2-coloring found
        return True

    # Start with all edges
    best_indices = list(range(len(all_edges)))
    best_count = len(best_indices)

    # Greedy Minimization
    # We maintain a list of indices of edges currently in the hypergraph.
    # We try to remove edges.
    
    # To speed up, we can keep a boolean array or a set of active indices.
    # But for small n, list of indices is fine.
    
    num_restarts = 50 # We have time budget, try many restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_indices = best_indices[:]
        random.shuffle(current_indices)
        
        # We iterate through the shuffled list of indices.
        # If an index is still in current_indices, we try to remove it.
        # To make removal O(1) or O(N) efficient, we can use a set for "active" and a list for "order".
        # But since we are modifying the list, let's just filter.
        
        # Actually, a better way for greedy removal:
        # active_set = set(current_indices)
        # for idx in current_indices:
        #   if idx not in active_set: continue
        #   test_set = active_set - {idx}
        #   if is_non_2_colorable(list(test_set)):
        #       active_set = test_set
        
        active_set = set(current_indices)
        
        for idx in current_indices:
            if time.time() > deadline:
                break
                
            if idx not in active_set:
                continue
                
            test_set = active_set - {idx}
            
            # Check if still non-2-colorable
            # Pass list of indices to is_non_2_colorable
            if is_non_2_colorable(list(test_set)):
                active_set = test_set
                
        if len(active_set) < best_count:
            best_count = len(active_set)
            best_indices = list(active_set)
            
            # If we reach a very low number, we might want to stop early?
            # The theoretical lower bound is 29.
            # If we hit 51, we might try to find 50.
            # Keep going until deadline.

    # Construct the final list of edges
    final_edges = [all_edges[i] for i in best_indices]
    
    return {'n': n, 'edges': final_edges}
