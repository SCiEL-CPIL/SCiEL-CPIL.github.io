import time

N = 40

def construct():
    start_time = time.time()
    
    # Base cycle edges
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
        
    # Chords from 0 to 2..20
    for d in range(2, 21):
        edges.add((0, d))
        
    edge_list = list(edges)
    
    # Generate cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycle = list(range(N))
        else:
            # Find d such that L = d+1 or L = 41-d
            # d = L-1 or d = 41-L
            # We need d in 2..20
            d1 = L - 1
            d2 = 41 - L
            
            if 2 <= d1 <= 20:
                # Cycle 0 -> 1 -> ... -> d1 -> 0
                cycle = list(range(d1 + 1))
                # Reverse to make it a simple list? Order doesn't matter for validity
                # But ensure it's a valid cycle in the graph
                # 0-1-...-d1-0
                # Edges: (0,1), (1,2)...(d1-1, d1), (d1, 0)
                # (d1, 0) is a chord if d1 > 1. Yes.
            elif 2 <= d2 <= 20:
                # Cycle 0 -> 39 -> ... -> d2 -> 0
                cycle = [0] + list(range(39, d2 - 1, -1))
                # 0-39-...-d2-0
                # Edges: (0,39), (39,38)...(d2+1, d2), (d2, 0)
                # (d2, 0) is a chord.
            else:
                # Should not happen
                cycle = list(range(L))
                
        cycles.append(cycle)
        
    return {'edges': edge_list, 'cycles': cycles}
