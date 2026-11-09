import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Base Hamiltonian cycle edges: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        
    # Chords from 0 to even k: 2, 4, 6, ..., 38
    K = list(range(2, 39, 2))
    for k in K:
        edges.add((0, k))
        
    edge_list = sorted(edges)
    edge_set = set(edge_list)
    K_set = set(K)
    
    # Pre-compute all cycles
    cycles = []
    
    for L in range(3, N+1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        elif L - 1 in K_set:
            # Direct: 0 -> 1 -> ... -> L-1 -> 0
            cycle = list(range(L))
        elif N - L + 1 in K_set:
            # Backwards: 0 -> N-1 -> N-2 -> ... -> (N-L+1) -> 0
            k = N - L + 1
            cycle = [0] + list(range(N-1, k-1, -1))
        else:
            # Find a, b in K with |a-b| = L-2
            diff = L - 2
            found = False
            for a in K:
                b1 = a + diff
                if b1 in K_set:
                    cycle = [0] + list(range(a, b1 + 1))
                    found = True
                    break
                b2 = a - diff
                if b2 in K_set:
                    cycle = [0] + list(range(b2, a + 1))
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find cycle for length {L}")
        
        cycles.append(cycle)
        
    # Verification
    for c in cycles:
        if len(set(c)) != len(c):
            raise ValueError(f"Duplicate vertices in cycle {c}")
        for i in range(len(c)):
            u, v = c[i], c[(i+1)%len(c)]
            if (min(u,v), max(u,v)) not in edge_set:
                raise ValueError(f"Missing edge ({u},{v}) in cycle {c}")

    return {'edges': edge_list, 'cycles': cycles}
