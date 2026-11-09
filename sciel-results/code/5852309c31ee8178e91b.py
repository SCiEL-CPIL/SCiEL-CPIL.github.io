import time

N = 40

def construct():
    deadline = time.time() + 85
    
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    for k in range(2, 21):
        edges.add((0, k))
    
    cycles = {}
    for L in range(3, N + 1):
        if L == N:
            cycles[L] = list(range(N))
        elif L <= 21:
            cycles[L] = list(range(0, L))
        else:
            k = 41 - L
            cycles[L] = [0] + list(range(k, N))
    
    def cycle_edges(cycle):
        n = len(cycle)
        return set((min(cycle[i], cycle[(i+1)%n]), max(cycle[i], cycle[(i+1)%n])) for i in range(n))
    
    def valid_cycle(cycle, edge_set):
        if len(set(cycle)) != len(cycle):
            return False
        return all(e in edge_set for e in cycle_edges(cycle))
    
    # Try to remove each edge
    for edge in list(edges):
        if time.time() > deadline:
            break
        edges.discard(edge)
        if all(valid_cycle(cycles[L], edges) for L in range(3, N+1)):
            pass  # Edge successfully removed
        else:
            edges.add(edge)  # Restore edge
    
    return {
        'edges': list(edges),
        'cycles': [cycles[L] for L in range(3, N+1)]
    }
