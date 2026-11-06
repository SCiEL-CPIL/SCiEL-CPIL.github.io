import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sparse pancyclic graph on N=40 vertices.
    Uses a Hamiltonian cycle plus chords from vertex 0 to cover all cycle lengths 3..40.
    """
    # 1. Generate base cycle edges
    # Vertices 0..39 in a ring
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # 2. Generate chords
    # Chords (0, k) for k in 2..20
    # These cover lengths:
    # k=2 -> len 3 (0-1-2-0) and len 39 (0-39-...-2-0)
    # k=3 -> len 4 and len 38
    # ...
    # k=20 -> len 21 and len 21
    chords = []
    for k in range(2, N // 2 + 1):
        edges.add((0, k))
        chords.append(k)
    
    # Convert set to list of tuples for output consistency
    edge_list = [tuple(sorted(e)) for e in edges]
    
    # 3. Construct cycles
    # We need one cycle of each length L from 3 to 40.
    cycles = []
    
    for L in range(3, N + 1):
        if L == N:
            # The main Hamiltonian cycle
            # Vertices 0, 1, ..., 39
            cyc = list(range(N))
        elif L <= N // 2 + 1:
            # Use the "short" path of chord (0, L-1)
            # k = L - 1
            # Path: 0 -> 1 -> ... -> k
            # Cycle: 0, 1, ..., k
            k = L - 1
            cyc = list(range(k + 1)) # 0 to k
            # The chord (0, k) closes it.
            # Note: The list represents the sequence of vertices.
            # Edges are (0,1), (1,2)...(k-1,k) and (k,0).
        else:
            # Use the "long" path of chord (0, k) where k = N + 1 - L
            # k is in range [2, N//2 - 1] for L in [N-1, N//2+2]
            # Actually, if L > 21, k = 41 - L.
            # If L=39, k=2. If L=22, k=19.
            k = N + 1 - L
            # Path: 0 -> N-1 -> N-2 -> ... -> k
            # Vertices: 0, N-1, N-2, ..., k
            # Count: 1 (for 0) + (N-1 - k + 1) = N - k + 1 = 40 - (41-L) + 1 = L.
            # Edges: (0, N-1), (N-1, N-2), ..., (k+1, k) and chord (k, 0).
            cyc = [0] + list(range(N - 1, k - 1, -1))
            
        cycles.append(cyc)
        
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END

if __name__ == '__main__':
    # Simple sanity check
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    for cyc in res['cycles']:
        L = len(cyc)
        if L < 3 or L > 40:
            raise ValueError(f"Invalid cycle length {L}")
        # Check distinct vertices
        if len(set(cyc)) != L:
            raise ValueError(f"Duplicate vertices in cycle of length {L}")
        # Check edges exist
        for i in range(L):
            u, v = cyc[i], cyc[(i + 1) % L]
            if (u, v) not in edges and (v, u) not in edges:
                raise ValueError(f"Missing edge ({u},{v}) in cycle of length {L}")
    print(f"Edges: {len(res['edges'])}, Cycles: {len(res['cycles'])}")
