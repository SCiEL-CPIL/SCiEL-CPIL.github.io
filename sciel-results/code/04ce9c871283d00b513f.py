"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Construct the Clebsch graph
    # Vertices are binary strings of length 5 with even parity
    # There are 16 such strings
    # Two vertices are connected if their Hamming distance is 2
    
    vertices = []
    for bits in itertools.product([0, 1], repeat=5):
        if sum(bits) % 2 == 0:
            vertices.append(bits)
            
    # Ensure we have exactly 16 vertices
    # 1 (00000) + 10 (weight 2) + 5 (weight 4) = 16
    
    edges = []
    n = len(vertices)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate Hamming distance
            dist = 0
            for k in range(5):
                if vertices[i][k] != vertices[j][k]:
                    dist += 1
            if dist == 2:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
