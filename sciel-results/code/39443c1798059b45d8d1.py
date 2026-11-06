import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Construction: 8 disjoint copies of K4
    # n = 32
    # Edges: 6 per K4, total 48
    # No C5 (K4 is K5-free and has no cycles > 3)
    # Independence number: 1 per K4, total 8. 8 < 9.
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
