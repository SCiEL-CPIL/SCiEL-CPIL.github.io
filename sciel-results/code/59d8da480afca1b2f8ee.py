import itertools
import time

def construct():
    n = 9
    r = 3
    # Generate all possible edges
    all_edges = list(itertools.combinations(range(n), r))
    
    # Pre-compute masks for 2-colorability
    # There are 2^9 = 512 possible 2-colorings.
    # We represent a coloring as an integer 0..511.
    # Bit i of the integer represents the color of vertex i (0 or 1).
    
    # For each edge, determine which colorings make it monochromatic.
    # We want a mask of colorings that are VALID (non-monochromatic) for the edge.
    
    masks = []
    for e in all_edges:
        u, v, w = e
        valid_mask = 0
        for c in range(512):
            cu = (c >> u) & 1
            cv = (c >> v) & 1
            cw = (c >> w) & 1
            # Monochromatic if all 0 or all 1
            if not (cu == cv == cw):
                valid_mask |= (1 << c)
        masks.append(valid_mask)
        
    # Search for minimum number of edges that is NOT 2-colorable
    # This means the intersection of valid masks is 0.
    
    best_edges = None
    best_colors = None
    
    # We know the seed has 10 edges. We only need to find < 10 to improve.
    # But let's search up to 10 to be safe, or just find the first one.
    # The minimum is likely small.
    
    deadline = time.time() + 40 # Safety margin
    
    for k in range(1, 11): # Check up to 10 edges
        if time.time() > deadline:
            break
            
        # Iterate through combinations of k edges
        # To speed up, we can break early if we find one
        for edges in itertools.combinations(range(len(all_edges)), k):
            if time.time() > deadline:
                break
                
            # Check if intersection of masks is 0
            # Start with all 1s (512 bits)
            current_mask = (1 << 512) - 1
            for idx in edges:
                current_mask &= masks[idx]
                if current_mask == 0:
                    # Found a non-2-colorable hypergraph
                    # Now find a valid 3-coloring
                    edge_list = [all_edges[idx] for idx in edges]
                    
                    # Find a 3-coloring
                    # Iterate 0..3^9 - 1
                    found_coloring = None
                    for c in range(3**9):
                        # Decode coloring
                        colors = []
                        temp_c = c
                        for i in range(n):
                            colors.append(temp_c % 3)
                            temp_c //= 3
                        
                        # Check if valid
                        valid = True
                        for e in edge_list:
                            cu = colors[e[0]]
                            cv = colors[e[1]]
                            cw = colors[e[2]]
                            if cu == cv == cw:
                                valid = False
                                break
                        
                        if valid:
                            found_coloring = colors
                            break
                    
                    if found_coloring is not None:
                        best_edges = [list(e) for e in edge_list]
                        best_colors = found_coloring
                        return {'edges': best_edges, 'colors': best_colors}
                    else:
                        # Should not happen for small k, but if it does, it's not 3-colorable
                        # We need exactly 3-chromatic. If it's 4-chromatic, it's not "exactly 3"?
                        # The task says "Minimize edge count among exactly three-chromatic triple systems".
                        # So we must ensure it IS 3-colorable.
                        pass
                        
    # If not found in search, fall back to seed or best found
    if best_edges is not None:
        return {'edges': best_edges, 'colors': best_colors}
        
    # Fallback to seed solution (10 edges)
    # Seed: combinations(range(5), 3) -> 10 edges. Colors: [0,0,1,1,2,0,0,0,0]
    # Let's verify the seed is valid.
    # Edges are all triples from {0,1,2,3,4}.
    # Colors: 0:0, 1:0, 2:1, 3:1, 4:2.
    # Edge (0,1,2): 0,0,1 -> OK
    # Edge (0,1,3): 0,0,1 -> OK
    # Edge (0,1,4): 0,0,2 -> OK
    # Edge (0,2,3): 0,1,1 -> OK
    # Edge (0,2,4): 0,1,2 -> OK
    # Edge (0,3,4): 0,1,2 -> OK
    # Edge (1,2,3): 0,1,1 -> OK
    # Edge (1,2,4): 0,1,2 -> OK
    # Edge (1,3,4): 0,1,2 -> OK
    # Edge (2,3,4): 1,1,2 -> OK
    # Is it 2-colorable?
    # If we restrict to 2 colors, can we color {0,1,2,3,4} such that no edge is mono?
    # This is the same as the hypergraph being 2-colorable.
    # The search above will determine if the seed is 2-colorable or not.
    # Actually, the seed program returns this, implying it's a valid candidate for the suite, but maybe it's 2-colorable?
    # If the seed is 2-colorable, it would be invalid for the "three-chromatic" requirement?
    # The task says: "The evaluator exhaustively rejects any hypergraph admitting a two-coloring."
    # So the seed MUST be non-2-colorable to have score 1.0.
    # My search should find something with <= 10 edges. If it finds < 10, score > 1.0.
    
    return {'edges': [list(e) for e in itertools.combinations(range(5), 3)], 'colors': [0,0,1,1,2,0,0,0,0]}
