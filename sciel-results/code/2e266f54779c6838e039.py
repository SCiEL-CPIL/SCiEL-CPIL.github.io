"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs the Fano plane (PG(2,2)) as a 3-uniform hypergraph on 9 vertices.
    The Fano plane is the smallest 3-uniform hypergraph that is not 2-colorable
    (i.e., it is 3-chromatic). It has 7 edges.
    Vertices 7 and 8 are isolated, satisfying the 9-vertex constraint.
    The number of edges (7) is the theoretical minimum for a 3-chromatic
    3-uniform hypergraph.
    """
    # The Fano plane can be constructed on vertices 1..7 using XOR.
    # Each edge is a triple {i, j, i^j} where i != j and i^j != 0.
    # We map these to 0..6.
    edges = []
    seen = set()
    
    # Generate edges for vertices 1 to 7
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = i ^ j
            # Ensure k is in the valid range and distinct from i, j
            if 1 <= k <= 7 and k != i and k != j:
                # Create a canonical sorted tuple to avoid duplicates
                triple = tuple(sorted((i - 1, j - 1, k - 1)))
                if triple not in seen:
                    seen.add(triple)
                    edges.append([triple[0], triple[1], triple[2]])
    
    # Sort edges for determinism
    edges.sort()
    
    # Verify we have exactly 7 edges (Fano plane property)
    # If not, something is wrong with the generation, but for Fano plane it is always 7.
    
    # Provide a valid 3-coloring.
    # The Fano plane is 3-colorable. A simple modular coloring works for this specific instance.
    # We assign colors 0, 1, 2 to vertices 0..8.
    # Vertices 7 and 8 are isolated, so their color doesn't affect edge constraints.
    colors = [v % 3 for v in range(9)]
    
    # Ensure types are Python int
    return {
        'edges': [[int(e[0]), int(e[1]), int(e[2])] for e in edges],
        'colors': [int(c) for c in colors]
    }

# EVOLVE-BLOCK-END
