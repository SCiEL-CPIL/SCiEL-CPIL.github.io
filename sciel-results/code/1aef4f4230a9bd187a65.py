import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Precompute edge masks for faster checking
    edge_masks = []
    for e in all_edges:
        m = 0
        for v in e:
            m |= (1 << v)
        edge_masks.append(m)
    
    def is_2_colorable(masks):
        # Check if there exists a coloring (0..2^n - 1) such that no edge is monochromatic
        for mask in range(1 << n):
            valid_coloring = True
            for em in masks:
                # Check if edge is monochromatic
                # If all bits in em are 0 in mask -> mono 0
                # If all bits in em are 1 in mask -> mono 1
                if (mask & em) == 0 or (mask & em) == em:
                    valid_coloring = False
                    break
            if valid_coloring:
                return True
        return False

    # Start with all edges
    current_masks = list(edge_masks)
    
    # Greedy removal of non-essential edges
    # An edge is essential if removing it makes the hypergraph 2-colorable
    # We want to keep the hypergraph non-2-colorable, so we remove edges only if it REMAINS non-2-colorable
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(current_masks)):
            if time.time() > deadline:
                break
            # Try removing edge i
            test_masks = current_masks[:i] + current_masks[i+1:]
            # If the remaining hypergraph is NOT 2-colorable, we can remove this edge
            if not is_2_colorable(test_masks):
                current_masks = test_masks
                changed = True
                break # Restart the loop with the new list
    
    # Convert back to list of lists
    # We need to map masks back to vertex lists
    # We can just store the indices of the original edges
    # Or reconstruct from masks
    # Since we started with all_edges, and only removed, we can track indices
    # But simpler: just convert the final masks back to vertex lists
    final_edges = []
    for em in current_masks:
        verts = []
        for v in range(n):
            if em & (1 << v):
                verts.append(v)
        final_edges.append(verts)
        
    return {'n': n, 'edges': final_edges}
