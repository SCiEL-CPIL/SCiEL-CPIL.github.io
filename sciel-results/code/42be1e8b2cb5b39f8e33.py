N = 40

def construct():
    edges = set()
    # Rim
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords
    K = list(range(2, 18)) + [22]
    for k in K:
        edges.add((0, k))
        
    edges = sorted(edges)
    
    cycles = []
    for length in range(3, N + 1):
        if length == N:
            cycles.append(list(range(N)))
        else:
            # Check singles
            found = False
            # L = k + 1 => k = L - 1
            k1 = length - 1
            if k1 in K:
                # Cycle 0-1-...-k1
                cycles.append(list(range(k1 + 1)))
                found = True
            # L = 41 - k => k = 41 - L
            k2 = 41 - length
            if not found and k2 in K:
                # Cycle 0-39-...-k2
                cyc = [0] + list(range(N - 1, k2 - 1, -1))
                cycles.append(cyc)
                found = True
                
            if not found:
                # Composite
                # L = 2 + (v - u) => v - u = L - 2
                diff = length - 2
                # Find u, v in K such that v - u = diff
                # We know u < v
                for u in K:
                    v = u + diff
                    if v in K:
                        # Cycle 0-u-(u+1)-...-v
                        cyc = [0] + list(range(u, v + 1))
                        cycles.append(cyc)
                        found = True
                        break
                if not found:
                    # Should not happen
                    raise ValueError(f"Cycle {length} not found")
            
    return {'edges': edges, 'cycles': cycles}
