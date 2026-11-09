import time

N = 40

def construct():
    # Seed construction: Star centered at 0 + Path 1-2-...-39
    # Edges: (0, i) for i in 1..39 AND (i, i+1) for i in 1..38
    edges = []
    for i in range(1, N):
        edges.append((0, i))
    for i in range(1, N - 1):
        edges.append((i, i + 1))
    
    # Cycles: 0-1-...-(k-1)-0 for k=3..40
    cycles = []
    for k in range(3, N + 1):
        cycles.append(list(range(k)))
        
    return {'edges': edges, 'cycles': cycles}
