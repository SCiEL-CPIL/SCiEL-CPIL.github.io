import time

def construct():
    # Start with the Groetzsch graph (Mycielski of C5) on vertices 0..10
    # V = 0..4, V' = 5..9, u = 10
    edges = []
    
    # Cycle 0-1-2-3-4-0
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        edges.append([u, v])
        
    # V' connections: i in V connected to (i+1)%5 and (i-1)%5 in V'
    # V' index is 5 + j
    for i in range(5):
        j1 = (i + 1) % 5
        j2 = (i - 1) % 5
        edges.append([i, 5 + j1])
        edges.append([i, 5 + j2])
        
    # u (10) connected to all V'
    for j in range(5):
        edges.append([10, 5 + j])
        
    # Add 5 new vertices 11..15
    # Connect each to all V' (5..9)
    for n in range(11, 16):
        for v_prime in range(5, 10):
            edges.append([n, v_prime])
            
    return edges
