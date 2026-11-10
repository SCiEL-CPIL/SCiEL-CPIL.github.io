"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring (no Property B).
    Uses brute-force 2-colorability check for small n, and random search for minimal edges.
    """
    import numpy as np
    import random
    import time
    from itertools import combinations
    
    def is_2_colorable_np(n, edges):
        """
        Check if a 5-uniform hypergraph on n vertices has a proper 2-colouring.
        Returns True if 2-colourable, False otherwise.
        Uses numpy for fast vectorized checking of all 2^n colorings.
        """
        if not edges:
            return True
            
        # Precompute all color masks
        masks = np.arange(1 << n)
        # colors[i, v] is 1 if vertex v is color 1 in mask i, else 0
        # Shape: (2^n, n)
        colors = (masks[:, None] >> np.arange(n)) & 1
        
        # valid_masks[i] is True if mask i satisfies all edges processed so far
        valid_masks = np.ones(len(masks), dtype=bool)
        
        for e in edges:
            # Sum of colors for vertices in edge e for all masks
            # e is a tuple of vertex indices
            s = colors[:, e].sum(axis=1)
            
            # A mask is invalid for this edge if the edge is monochromatic
            # i.e., sum is 0 (all 0s) or sum is 5 (all 1s)
            invalid = (s == 0) | (s == 5)
            
            valid_masks &= ~invalid
            
            # Early termination: if no valid masks left, it's not 2-colorable
            if not np.any(valid_masks):
                return False
        
        # If we reach here, check if any mask is still valid
        return np.any(valid_masks)
    
    start_time = time.time()
    time_limit = 110
    
    # We search for n in 9, 10, 11
    # For each n, we try random shuffles of the complete hypergraph edges
    # and look for the smallest prefix that is not 2-colorable.
    
    for n in [9, 10, 11]:
        if time.time() - start_time > time_limit:
            break
            
        all_edges = list(combinations(range(n), 5))
        
        # Number of trials for random shuffles
        # We can afford many trials since the check is fast
        max_trials = 5000
        
        for trial in range(max_trials):
            if time.time() - start_time > time_limit:
                break
                
            random.shuffle(all_edges)
            
            # Try increasing k (number of edges)
            # Start from a reasonable lower bound to find minimal edges quickly
            # If we start too low, we waste time checking 2-colorable instances.
            # But we want minimal edges.
            # Let's try k from 40 to 60.
            # If we find one, we return.
            
            # To be more efficient, we can binary search the threshold?
            # No, we just want to find ANY non-2-colorable one with few edges.
            # Scanning up from 40 is good.
            
            for k in range(40, min(len(all_edges), 60)):
                if time.time() - start_time > time_limit:
                    break
                
                edges_subset = all_edges[:k]
                
                if not is_2_colorable_np(n, edges_subset):
                    # Found a non-2-colorable hypergraph with k edges
                    return {'n': n, 'edges': [list(e) for e in edges_subset]}
    
    # Fallback: return the complete hypergraph on 9 vertices (126 edges)
    # This is guaranteed to be non-2-colourable
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
