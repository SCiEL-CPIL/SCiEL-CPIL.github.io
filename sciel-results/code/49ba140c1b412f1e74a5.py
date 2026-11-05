import itertools
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 9
    vertices = range(n)
    
    # Generate all 5-uniform edges
    all_edges = [list(e) for e in itertools.combinations(vertices, 5)]
    
    # Convert to bitmasks for faster checking
    def edge_to_mask(edge):
        mask = 0
        for v in edge:
            mask |= (1 << v)
        return mask
        
    current_masks = [edge_to_mask(e) for e in all_edges]
    
    # Cache for is_not_2_colorable? 
    # The state is a tuple of masks. Too many states.
    # But n is small.
    
    def is_not_2_colorable(masks):
        # Check if there exists a 2-coloring of 9 vertices with no monochromatic edge
        # 2^9 = 512 colorings
        for c in range(512):
            is_valid_coloring = True
            for m in masks:
                # Edge is monochromatic if all bits are 0 or all bits are 1
                if (c & m) == 0 or (c & m) == m:
                    is_valid_coloring = False
                    break
            if is_valid_coloring:
                return False # Found a valid 2-coloring, so it IS 2-colorable
        return True # No valid 2-coloring, so it is NOT 2-colorable

    # Greedy removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(current_masks)
        for i in range(len(current_masks)):
            if time.time() > deadline:
                break
            
            mask_to_remove = current_masks[i]
            # Create new list without mask_to_remove
            new_masks = current_masks[:i] + current_masks[i+1:]
            
            # Check if the reduced hypergraph is still not 2-colorable
            if is_not_2_colorable(new_masks):
                current_masks = new_masks
                improved = True
                break # Restart the loop with the new list
        
    # Convert masks back to lists of vertices
    final_edges = []
    for m in current_masks:
        edge = []
        for v in range(n):
            if m & (1 << v):
                edge.append(v)
        final_edges.append(edge)
        
    return {'n': n, 'edges': final_edges}
