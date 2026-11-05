def construct():
    N = 40
    # 1. Create a Hamiltonian Cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        u = i
        v = (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
    
    # 2. Add Chords from Vertex 0 to Vertices 2..20
    # Chord (0, v) creates cycles of length v+1 and 41-v
    # We need to cover lengths 3..39
    # v=2 -> 3, 39
    # v=3 -> 4, 38
    # ...
    # v=19 -> 20, 22
    # v=20 -> 21, 21
    # This covers 3..39 completely.
    for v in range(2, 21):
        u = 0
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    edge_list = list(edges)
    
    # 3. Construct Cycles
    # Cycle of length L is a list of L distinct vertices v0..vL-1
    # such that (v0,v1), ..., (vL-2,vL-1), (vL-1,v0) are edges.
    cycles = []
    
    # Helper to check if edge exists
    def has_edge(u, v):
        if u > v: u, v = v, u
        return (u, v) in edges

    # For each length L from 3 to 40
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle: 0, 1, ..., 39
            cycle = list(range(N))
            cycles.append(cycle)
        else:
            # Find a chord (0, v) that generates length L
            # L = v + 1  => v = L - 1
            # L = 41 - v => v = 41 - L
            
            v1 = L - 1
            v2 = 41 - L
            
            found = False
            # Try v1
            if 2 <= v1 <= 20:
                # Cycle: 0, 1, ..., v1
                # Vertices: 0, 1, ..., v1
                # Length is v1 + 1 = L
                cycle = list(range(v1 + 1))
                # Verify edges
                valid = True
                for i in range(L - 1):
                    if not has_edge(cycle[i], cycle[i+1]):
                        valid = False
                        break
                if has_edge(cycle[-1], cycle[0]):
                    if valid:
                        cycles.append(cycle)
                        found = True
                else:
                    valid = False
            
            # Try v2 if v1 didn't work or wasn't in range
            if not found and 2 <= v2 <= 20:
                # Cycle: 0, 39, 38, ..., v2
                # Vertices: 0, 39, 38, ..., v2
                # Length is 41 - v2 = L
                # Sequence: 0, 39, 38, ..., v2
                # Note: The path is 0 -> 39 -> 38 ... -> v2 -> 0
                cycle = [0] + list(range(39, v2 - 1, -1))
                # Verify edges
                valid = True
                for i in range(L - 1):
                    if not has_edge(cycle[i], cycle[i+1]):
                        valid = False
                        break
                if has_edge(cycle[-1], cycle[0]):
                    if valid:
                        cycles.append(cycle)
                        found = True
                else:
                    valid = False
            
            # If still not found (should not happen), fallback to a generic search
            # But based on math, v1 or v2 must be in 2..20 for L in 3..39
            # L=3: v1=2 (ok), v2=38 (no)
            # L=20: v1=19 (ok), v2=21 (no)
            # L=21: v1=20 (ok), v2=20 (ok)
            # L=39: v1=38 (no), v2=2 (ok)
            
            if not found:
                # This should never trigger for L in 3..39 with the chosen chords
                # But to be safe, we could do a simple BFS/DFS for a cycle of length L
                # However, the construction is proven, so we just raise an error or return empty
                # To satisfy the interface, we must return a valid cycle.
                # We'll use a dummy cycle if logic fails, but it shouldn't.
                pass

    return {'edges': edge_list, 'cycles': cycles}

if __name__ == '__main__':
    res = construct()
    print(f"Edges: {len(res['edges'])}")
    print(f"Cycles: {len(res['cycles'])}")
    # Quick validation
    edges_set = set(tuple(sorted(e)) for e in res['edges'])
    for i, c in enumerate(res['cycles']):
        L = i + 3
        if len(c) != L:
            print(f"Cycle {L} has wrong length {len(c)}")
            break
        if len(set(c)) != L:
            print(f"Cycle {L} has duplicates")
            break
        for j in range(L):
            u, v = c[j], c[(j+1)%L]
            if u > v: u, v = v, u
            if (u, v) not in edges_set:
                print(f"Cycle {L} missing edge ({u}, {v})")
                break
    else:
        print("All cycles valid.")
