import time
from itertools import combinations

def construct():
    """
    Constructs a 5-uniform hypergraph with no proper 2-coloring.
    Uses a greedy set cover approach on n=11 vertices to minimize the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Strategy:
    # 1. Fix n=11.
    # 2. The universe is the set of all 2^11 colorings.
    # 3. Each edge "covers" (violates) the colorings where it is monochromatic.
    # 4. Greedily select edges to cover the entire universe with minimum edges.
    
    n = 11
    vertices = list(range(n))
    all_edges = list(combinations(vertices, 5))
    
    num_colorings = 1 << n
    # Bitset of uncovered colorings. Bit i is 1 if coloring i is not yet violated.
    uncovered = (1 << num_colorings) - 1
    
    # Precompute violation masks for each edge
    edge_masks = []
    for e in all_edges:
        edge_vertex_mask = 0
        for v in e:
            edge_vertex_mask |= (1 << v)
        
        mask = 0
        # Iterate all colorings to see which ones are monochromatic on this edge
        # Optimization: A coloring c is monochromatic on E iff 
        # (c & edge_vertex_mask) == 0 OR (c & edge_vertex_mask) == edge_vertex_mask
        for c in range(num_colorings):
            val = c & edge_vertex_mask
            if val == 0 or val == edge_vertex_mask:
                mask |= (1 << c)
        edge_masks.append(mask)
        
        if time.time() > deadline:
            break

    selected_edges = []
    
    # Greedy Set Cover
    while uncovered != 0:
        best_edge_idx = -1
        best_gain = 0
        
        for i, mask in enumerate(edge_masks):
            # Calculate how many new colorings this edge covers
            gain_mask = mask & uncovered
            if gain_mask:
                # Use bit_count for efficiency
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_edge_idx = i
        
        if best_edge_idx == -1:
            # No edge covers any remaining coloring (should not happen if n is large enough)
            break
            
        selected_edges.append(all_edges[best_edge_idx])
        uncovered &= ~edge_masks[best_edge_idx]
        
        if time.time() > deadline:
            break

    # Verify with backtracking
    if selected_edges and not _is_2_colorable(n, selected_edges):
        return {'n': n, 'edges': [list(e) for e in selected_edges]}
    
    # Fallback: Complete hypergraph on 9 vertices (126 edges)
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

def _is_2_colorable(n, edges):
    """
    Checks if a 2-coloring exists using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    color = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = -1
                for v in e:
                    if color[v] == 0:
                        c0 += 1
                    elif color[v] == 1:
                        c1 += 1
                    else:
                        unassigned = v
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if unassigned != -1:
                    if c0 == 4:
                        if color[unassigned] != 1:
                            color[unassigned] = 1
                            changed = True
                    elif c1 == 4:
                        if color[unassigned] != 0:
                            color[unassigned] = 0
                            changed = True
        return True

    def backtrack():
        if not propagate():
            return False
            
        # Find first unassigned vertex
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        
        if v == -1:
            return True
        
        # Try 0
        color[v] = 0
        if backtrack():
            return True
            
        # Try 1
        color[v] = 1
        if backtrack():
            return True
            
        color[v] = -1
        return False

    if not propagate():
        return False
        
    return backtrack()

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    # Move logic here to keep construct clean? 
    # No, the prompt says "Keep the function names... of the seed program".
    # The seed program has `construct` doing the work.
    # I will just put the work in construct.
    pass
