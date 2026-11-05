import time

def construct():
    n = 16
    edges = []
    
    # Generate Clebsch graph
    # Vertices 0..15 (4-bit integers)
    # Adjacency: Hamming distance 1 or 4
    
    for u in range(n):
        for v in range(u + 1, n):
            xor_val = u ^ v
            # Count bits
            # Python 3.x: int.bit_count()
            # For compatibility, use bin().count('1')
            weight = bin(xor_val).count('1')
            if weight == 1 or weight == 4:
                edges.append([u, v])
                
    return edges
