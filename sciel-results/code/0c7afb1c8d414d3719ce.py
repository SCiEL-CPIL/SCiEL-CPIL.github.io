import numpy as np
from itertools import combinations
import time

def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring, minimizing edges.
    
    Approach: 
    1. Model as SAT: Vertex i is bit i. Edge e is monochromatic if (color & mask_e) == 0 or (color & mask_e) == mask_e.
    2. Greedy: Start with all 2^n colorings valid. Iteratively pick the edge that eliminates the most valid colorings.
    3. Use NumPy for vectorized checks to speed up the process.
    4. Try small n (9 to 13) and return the hypergraph with the fewest edges.
    """
    
    def construct_for_n(n):
        # Generate all possible edges
        all_edges = list(combinations(range(n), 5))
        # Precompute masks for all edges
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            
        # Initialize valid colorings as a NumPy array of all 2^n integers
        valid_colorings = np.arange(1 << n, dtype=np.int64)
        
        selected_edges = []
        used_edge_indices = set()
        
        # We will iterate until valid_colorings is empty or no more edges can be added
        while len(valid_colorings) > 0:
            best_edge_idx = -1
            best_count = 0
            
            # For each unused edge, count how many valid colorings it invalidates
            for i, mask in enumerate(edge_masks):
                if i in used_edge_indices:
                    continue
                
                # Compute intersection for all valid colorings
                inter = valid_colorings & mask
                is_mono = (inter == 0) | (inter == mask)
                count = np.sum(is_mono)
                
                if count > best_count:
                    best_count = count
                    best_edge_idx = i
                    
            if best_edge_idx == -1 or best_count == 0:
                # No edge can reduce the solution space further
                break
                
            used_edge_indices.add(best_edge_idx)
            selected_edges.append(list(all_edges[best_edge_idx]))
            
            # Update valid_colorings: keep only those that are NOT monochromatic on the chosen edge
            mask = edge_masks[best_edge_idx]
            inter = valid_colorings & mask
            is_mono = (inter == 0) | (inter == mask)
            valid_colorings = valid_colorings[~is_mono]
            
        if len(valid_colorings) == 0:
            return {'n': n, 'edges': selected_edges}
        else:
            return None

    best_result = None
    min_edges = float('inf')
    
    # Try n from small to larger
    # We check a range and pick the best (fewest edges).
    for n in range(9, 14):
        res = construct_for_n(n)
        if res:
            num_edges = len(res['edges'])
            if num_edges < min_edges:
                min_edges = num_edges
                best_result = res

    if best_result:
        return best_result
        
    # Fallback: Return a known valid hypergraph if greedy fails or is too slow
    # A complete graph K_9^5 has 126 edges. Definitely not 2-colorable.
    # But we want few edges. 
    # If greedy failed, we return a safe but potentially suboptimal result.
    # To ensure validity, we can use a known small non-2-colorable hypergraph if available, 
    # or just return the first one found by a less optimized method.
    # Given the time constraint, if the NumPy version fails, we return a dense one.
    
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
