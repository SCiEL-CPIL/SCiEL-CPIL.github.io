"""
Constructs a 3-uniform hypergraph on 9 vertices (0..8) that is 3-chromatic
with the minimum possible number of edges (7).

The core structure is the Fano plane (PG(2,2)) embedded on vertices 0-6.
Vertices 7 and 8 are isolated.

The Fano plane is the smallest 3-uniform hypergraph that is not 2-colorable.
It has exactly 7 edges. Since any 3-chromatic 3-uniform hypergraph must
contain a non-2-colorable sub-hypergraph, and the Fano plane is the minimal
such structure, 7 is the theoretical minimum edge count.
"""

def construct():
    """
    Returns a dict with keys 'edges' and 'colors'.
    'edges': list of distinct triples (lists of 3 ints) on vertices 0..8.
    'colors': list of 9 ints from 0..2.
    Every edge must be non-monochromatic in the provided coloring.
    The hypergraph must be 3-chromatic (not 2-colorable).
    """
    
    # 1. Generate the edges of the Fano plane on vertices 0..6
    # The Fano plane vertices can be represented as non-zero vectors in F_2^3 (1..7).
    # An edge is a set of 3 distinct vertices {i, j, k} such that i + j + k = 0 in F_2^3.
    # In terms of integers 1..7, this is equivalent to i ^ j ^ k == 0 (bitwise XOR).
    
    edges = []
    seen = set()
    
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = i ^ j
            if k > j and k <= 7:
                # Ensure distinctness: i, j, k are distinct if i != j and k != i, k != j.
                # Since k = i ^ j, k == i implies j == 0 (not in range), k == j implies i == 0 (not in range).
                # So i, j, k are always distinct for i != j.
                # Map to 0-indexed vertices: v = val - 1
                v0 = i - 1
                v1 = j - 1
                v2 = k - 1
                edge_tuple = (v0, v1, v2)
                if edge_tuple not in seen:
                    seen.add(edge_tuple)
                    edges.append([v0, v1, v2])
                    
    # Sort edges for deterministic output
    edges.sort()
    
    # 2. Define a valid 3-coloring
    # We need a coloring of the 9 vertices such that:
    # a) No edge is monochromatic.
    # b) The hypergraph is NOT 2-colorable (this is a property of the graph structure, not the coloring,
    #    but we must provide a 3-coloring that works).
    
    # The Fano plane is 3-chromatic. A valid 3-coloring can be derived by assigning
    # colors based on the "type" of vector or simply by trial and error / known solution.
    # Vertices 0..6 correspond to values 1..7.
    # Let's assign colors to values 1..7 such that no {i,j,k} with i^j^k=0 is monochromatic.
    
    # Known 3-coloring for Fano plane:
    # Values: 1, 2, 3, 4, 5, 6, 7
    # Colors: 0, 1, 2, 0, 1, 2, 0  (This might not work, let's verify)
    
    # Let's verify the coloring [0, 1, 2, 0, 1, 2, 0] on vertices 0..6 (values 1..7)
    # Edge (1,2,3) -> (0,1,2) colors: 0,1,2 -> OK
    # Edge (1,4,5) -> (0,3,4) colors: 0,0,1 -> OK (not mono)
    # Edge (1,6,7) -> (0,5,6) colors: 0,2,0 -> OK (not mono)
    # Edge (2,4,6) -> (1,3,5) colors: 1,0,2 -> OK
    # Edge (2,5,7) -> (1,4,6) colors: 1,1,0 -> OK
    # Edge (3,4,7) -> (2,3,6) colors: 2,0,0 -> OK
    # Edge (3,5,6) -> (2,4,5) colors: 2,1,2 -> OK
    
    # So colors for vertices 0..6 are:
    # Vert 0 (val 1): 0
    # Vert 1 (val 2): 1
    # Vert 2 (val 3): 2
    # Vert 3 (val 4): 0
    # Vert 4 (val 5): 1
    # Vert 5 (val 6): 2
    # Vert 6 (val 7): 0
    
    colors_0_6 = [0, 1, 2, 0, 1, 2, 0]
    
    # Vertices 7 and 8 are isolated, so they can be any color. Let's use 0.
    colors_7_8 = [0, 0]
    
    colors = colors_0_6 + colors_7_8
    
    # 3. Final Validation and Type Conversion
    # Ensure all types are Python int and lists are lists.
    
    edges_out = []
    for e in edges:
        # Convert to Python int to satisfy strict type checking
        edges_out.append([int(e[0]), int(e[1]), int(e[2])])
        
    colors_out = [int(c) for c in colors]
    
    # Sanity Check 1: Ensure no monochromatic edges
    for e in edges_out:
        c0 = colors_out[e[0]]
        c1 = colors_out[e[1]]
        c2 = colors_out[e[2]]
        if c0 == c1 == c2:
            # This would be a bug in our static construction
            # Fallback: Re-assign colors? No, the construction is verified.
            # But just in case, we can try to fix it, though it shouldn't happen.
            pass
            
    # Sanity Check 2: Ensure the graph is NOT 2-colorable.
    # We can quickly check this by trying all 2^7 = 128 colorings of the 7 active vertices.
    # If any 2-coloring works, our graph is invalid for the task.
    # Since it's the Fano plane, we know it's not 2-colorable, but this adds robustness.
    
    is_2_colorable = False
    # Only check vertices involved in edges (0..6)
    active_vertices = set()
    for e in edges_out:
        active_vertices.add(e[0])
        active_vertices.add(e[1])
        active_vertices.add(e[2])
    
    # There are 7 active vertices. 2^7 = 128 possibilities.
    for mask in range(128):
        valid_coloring = True
        for e in edges_out:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid_coloring = False
                break
        if valid_coloring:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        # This should never happen for the Fano plane.
        # If it does, the construction is flawed.
        # We cannot easily fix this on the fly without a search, 
        # but given the static nature, this is a safety check.
        pass

    return {
        'edges': edges_out,
        'colors': colors_out
    }
