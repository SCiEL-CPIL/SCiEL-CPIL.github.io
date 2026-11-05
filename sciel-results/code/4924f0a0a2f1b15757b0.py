import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    edges = []
    
    # Clebsch graph construction
    # Vertices 0..15 (4-bit vectors)
    # Adjacency: Hamming distance 1 or 4
    
    for u in range(n):
        for v in range(u + 1, n):
            xor_val = u ^ v
            # Check Hamming distance
            # popcount in Python
            dist = bin(xor_val).count('1')
            if dist == 1 or dist == 4:
                edges.append([u, v])
                
    # The Clebsch graph is triangle-free and has chromatic number 4.
    # It has 40 edges.
    
    # Verification step (optional but good for safety if logic was complex, here it's deterministic)
    # We can quickly verify triangle-free property in O(E * n) or similar if needed, 
    # but the construction is mathematically sound.
    
    return edges
