N = 40

def construct():
    # Build the 40-cycle rim (40 edges)
    edges_set = set()
    for i in range(N):
        a, b = i, (i + 1) % N
        if a > b:
            a, b = b, a
        edges_set.add((a, b))
    
    # 16 chords from vertex 0 that cover all cycle lengths 3..39
    # via single-chord (k+1 or 41-k) or double-chord (2+|a-b|) cycles
    K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24]
    K_set = set(K)
    
    for k in K:
        a, b = 0, k
        if a > b:
            a, b = b, a
        edges_set.add((a, b))
    
    edges = sorted(edges_set)
    
    # Build cycle certificates for lengths 3..40
    cycles = []
    
    for length in range(3, N + 1):
        if length == N:
            cycles.append(list(range(N)))
            continue
        
        found = False
        
        # Try single chord: L = k+1 => k = L-1
        k1 = length - 1
        if k1 in K_set:
            cycles.append(list(range(k1 + 1)))
            found = True
        
        # Try single chord: L = 41-k => k = 41-L
        if not found:
            k2 = 41 - length
            if k2 in K_set:
                cyc = [0] + list(range(N - 1, k2 - 1, -1))
                cycles.append(cyc)
                found = True
        
        # Try double chord: L = 2 + |k1 - k2| => |k1-k2| = L-2
        if not found:
            diff = length - 2
            for u in K:
                v1 = u + diff
                v2 = u - diff
                if v1 in K_set and v1 != u:
                    # Cycle 0-u-(u+1)-...-v1-0
                    cyc = [0] + list(range(u, v1 + 1))
                    cycles.append(cyc)
                    found = True
                    break
                elif v2 in K_set and v2 != u:
                    # Cycle 0-v2-(v2+1)-...-u-0
                    cyc = [0] + list(range(v2, u + 1))
                    cycles.append(cyc)
                    found = True
                    break
        
        if not found:
            raise ValueError(f"Cycle {length} not found")
    
    return {'edges': edges, 'cycles': cycles}
