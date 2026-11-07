"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START

def construct():
    # The minimum 3-chromatic 3-uniform hypergraph is the Fano plane.
    # It has 7 vertices and 7 edges.
    # We embed it in the 9-vertex space (0..8) by using vertices 0..6 
    # for the Fano plane and leaving 7 and 8 isolated.
    
    # Construct the Fano plane edges algebraically:
    # The Fano plane can be represented as lines in the projective plane over GF(2).
    # Points are non-zero vectors in GF(2)^3. Lines are sets of 3 points that sum to 0.
    
    # Vertices in GF(2)^3 (excluding 0)
    points = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1)
    ]
    
    # Map point tuples to indices 0..6
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    edges = []
    
    # Generate all lines (triples of points that sum to zero mod 2)
    for i in range(7):
        for j in range(i + 1, 7):
            # Calculate third point required to sum to 0
            # p_i + p_j + p_k = 0 => p_k = p_i + p_j (in GF(2))
            p_i = points[i]
            p_j = points[j]
            p_k = (
                (p_i[0] + p_j[0]) % 2,
                (p_i[1] + p_j[1]) % 2,
                (p_i[2] + p_j[2]) % 2
            )
            
            # Check if p_k is a valid distinct point
            if p_k in point_to_idx:
                k = point_to_idx[p_k]
                if k > j:  # Ensure unique edges, sorted indices
                    edges.append([i, j, k])
    
    # Verify we have exactly 7 edges
    # The Fano plane has 7 lines.
    # In the loop above, for each pair (i,j), we find k.
    # Since each line has 3 points, each line is counted 3 times if we don't restrict.
    # By requiring i < j < k, each line is counted exactly once.
    
    # Sort edges for determinism
    edges.sort()
    
    # Convert to Python ints and lists
    edges_out = [[int(v) for v in e] for e in edges]
    
    # Provide a valid 3-coloring.
    # We need to color vertices 0..6 such that no edge is monochromatic.
    # And vertices 7,8 can be any color, let's say 0.
    
    # A known 3-coloring for the Fano plane (vertices 0-6):
    # Let's assign colors based on the binary representation or just a known pattern.
    # Pattern:
    # 0: 001 -> color 0
    # 1: 010 -> color 1
    # 2: 001? No, indices are arbitrary.
    # Let's use the specific mapping from points above:
    # 0: (1,0,0)
    # 1: (0,1,0)
    # 2: (0,0,1)
    # 3: (1,1,0)
    # 4: (1,0,1)
    # 5: (0,1,1)
    # 6: (1,1,1)
    
    # Color assignment:
    # We can simply check if a coloring works.
    # Try colors: [0, 1, 2, 1, 2, 0, 1]
    colors_0_6 = [0, 1, 2, 1, 2, 0, 1]
    
    # Verify non-monochromatic for all edges
    valid = True
    for e in edges:
        c0 = colors_0_6[e[0]]
        c1 = colors_0_6[e[1]]
        c2 = colors_0_6[e[2]]
        if c0 == c1 == c2:
            valid = False
            break
    
    if not valid:
        # Fallback to another valid coloring if the above fails (it shouldn't)
        # But the Fano plane is 3-chromatic, so many colorings work.
        # Let's try to find one programmatically if needed, but hardcoding a verified one is safer.
        # The above coloring is verified to work for the standard Fano embedding.
        pass
        
    # Add colors for isolated vertices 7 and 8
    colors = colors_0_6 + [0, 0]
    
    # Final type checks
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}

# EVOLVE-BLOCK-END
