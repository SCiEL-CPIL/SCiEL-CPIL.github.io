import time

def construct():
    # The Clebsch graph is a triangle-free graph on 16 vertices
    # with chromatic number 4 and 40 edges.
    # It is defined on the 16 binary strings of length 5 with the first bit 0.
    # Two vertices are adjacent if their Hamming distance is 1 or 4.
    
    n = 16
    # Generate the 16 vertices
    # We use integers 0..15 representing the last 4 bits, with the first bit implicitly 0.
    # So vertex i corresponds to binary string 0 b3 b2 b1 b0.
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate Hamming distance between i and j (4 bits)
            # Since the first bit is 0 for both, the distance is just the popcount of i XOR j
            xor_val = i ^ j
            dist = bin(xor_val).count('1')
            
            # Adjacency condition for Clebsch graph: distance 1 or 4
            if dist == 1 or dist == 4:
                edges.append([i, j])
                
    return edges
