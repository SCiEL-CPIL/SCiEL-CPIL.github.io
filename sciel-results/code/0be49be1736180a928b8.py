"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Construct a triangle-free graph with no independent set of size 10.
    # We use the blow-up of the Clebsch graph.
    # The Clebsch graph C_16 has 16 vertices, is triangle-free, and has alpha=2.
    # We replace each vertex with an independent set of size k=2.
    # Each edge (u, v) in C_16 becomes a K_{2,2} between the two independent sets.
    # The resulting graph has n = 16 * 2 = 32 vertices.
    # It is triangle-free because C_16 is triangle-free.
    # Its independence number is alpha = 2 * alpha(C_16) = 2 * 2 = 4.
    # Since 4 < 10, the condition is satisfied.
    
    n = 32
    edges = []
    
    # Define the Clebsch graph edges.
    # The Clebsch graph can be defined as the graph on vertices {0, 1, 2, 3} x {0, 1, 2, 3}
    # where (a,b) is adjacent to (c,d) if a!=c and b!=d? No, that's the Rook graph.
    # The Clebsch graph is the halved 5-cube.
    # Vertices are 5-bit strings with even parity (16 vertices).
    # Two vertices are adjacent if they differ in exactly 2 positions.
    
    # Let's generate the Clebsch graph vertices and edges.
    clebsch_vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            clebsch_vertices.append(i)
    
    # Map clebsch vertices to indices 0..15
    idx_map = {v: i for i, v in enumerate(clebsch_vertices)}
    
    clebsch_edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            v1 = clebsch_vertices[i]
            v2 = clebsch_vertices[j]
            # Check if they differ in exactly 2 bits
            diff = v1 ^ v2
            if bin(diff).count('1') == 2:
                clebsch_edges.append((i, j))
    
    # Now perform the blow-up with k=2
    # Vertex i in C_16 becomes vertices 2*i and 2*i+1 in the new graph.
    # No edges between 2*i and 2*i+1.
    # For each edge (u, v) in C_16, add all 4 edges between {2u, 2u+1} and {2v, 2v+1}.
    
    for u, v in clebsch_edges:
        u1, u2 = 2 * u, 2 * u + 1
        v1, v2 = 2 * v, 2 * v + 1
        edges.append([u1, v1])
        edges.append([u1, v2])
        edges.append([u2, v1])
        edges.append([u2, v2])
        
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
