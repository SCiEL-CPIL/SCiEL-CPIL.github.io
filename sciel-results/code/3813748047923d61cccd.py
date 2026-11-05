import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    # Base cycle edges
    edges = [(i, i + 1) for i in range(N - 1)]
    edges.append((N - 1, 0))
    
    # Chords from 0 to even vertices 2, 4, ..., N-2
    for k in range(2, N, 2):
        edges.append((0, k))
        
    # Ensure unique edges (though construction is already unique)
    # Set of edges for quick lookup if needed, but we just return the list
    # The task requires 'edges' to be undirected edge pairs.
    # Standard format is usually sorted tuples or just pairs.
    # Let's normalize to (min(u,v), max(u,v))
    normalized_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        normalized_edges.add((u, v))
    
    final_edges = list(normalized_edges)
    
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        elif k % 2 == 1:
            # Odd length k: Cycle on vertices 0, 1, ..., k-1
            # Edges: (0,1)...(k-2,k-1), (k-1,0)
            # k-1 is even, so (k-1,0) is a chord.
            cycle = list(range(k))
        else:
            # Even length k: Cycle on vertices 0, 2, 3, ..., k
            # Edges: (0,2), (2,3)...(k-1,k), (k,0)
            # k is even, so (k,0) is a chord. (0,2) is a chord.
            # Vertices: 0, 2, 3, ..., k
            cycle = [0] + list(range(2, k + 1))
        cycles.append(cycle)
        
    return {'edges': final_edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
