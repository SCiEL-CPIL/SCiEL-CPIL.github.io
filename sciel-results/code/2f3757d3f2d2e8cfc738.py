def _construct_cage_320(k, g, deadline):
    if k != 4 or g != 10:
        return None, []
    
    # Base graph: 8 vertices, 4-regular
    # Vertices 0..7
    # Edges: (i, j) if i != j and i != (j+4)%8
    base_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if i != (j+4)%8 and j != (i+4)%8:
                base_edges.append((i, j))
                
    # Check base is 4-regular
    deg = [0]*8
    for u, v in base_edges:
        deg[u]+=1
        deg[v]+=1
    if not all(d==4 for d in deg):
        return None, []
        
    # Try random voltages in Z_40
    import random
    random.seed(42)
    
    # Get all cycles in base up to length 6
    # For 8 vertices, we can just check all paths of length 3, 4, 5, 6
    # But easier: check all subsets? No.
    # We need to check if there exists a cycle with "bad" voltage sum.
    
    # Precompute all simple cycles in base
    cycles = []
    # DFS to find cycles
    def find_cycles(start, current, path, visited):
        if len(path) > 1 and current == start:
            cycles.append(path[:])
            return
        for nxt in range(8):
            if (current, nxt) in base_edges or (nxt, current) in base_edges:
                if nxt == start and len(path) >= 3:
                    cycles.append(path + [nxt])
                elif nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    find_cycles(start, nxt, path, visited)
                    path.pop()
                    visited.remove(nxt)
                    
    for i in range(8):
        find_cycles(i, i, [i], {i})
        
    # Remove duplicates and reverse
    unique_cycles = []
    seen = set()
    for c in cycles:
        # Normalize
        min_idx = c.index(min(c))
        norm = tuple(c[min_idx:] + c[:min_idx])
        # Check reverse
        rev = tuple(norm[::-1])
        if norm not in seen and rev not in seen:
            unique_cycles.append(norm)
            seen.add(norm)
            seen.add(rev)
            
    # Filter short cycles (length < 10)
    short_cycles = [c for c in unique_cycles if len(c) < 10]
    
    # Try to assign voltages
    # We need sum(v_e) for each cycle to satisfy gcd condition
    # This is a hard CSP. Try random assignments and check.
    
    for attempt in range(1000):
        if _time.time() > deadline - 5.0:
            break
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, 39)
            voltages[(v, u)] = (-voltages[(u, v)]) % 40
            
        valid = True
        for c in short_cycles:
            s = 0
            for i in range(len(c)-1):
                u, v = c[i], c[i+1]
                s = (s + voltages[(u, v)]) % 40
            # Check gcd
            g_val = _np.gcd(s, 40) if s != 0 else 40
            L = len(c)
            cover_len = L * (40 // g_val)
            if cover_len < 10:
                valid = False
                break
                
        if valid:
            # Construct cover
            n = 320
            edges = []
            for (u, v), vol in voltages.items():
                if u > v: continue # Only one direction
                for i in range(40):
                    u_cover = u * 40 + i
                    v_cover = v * 40 + (i + vol) % 40
                    edges.append((u_cover, v_cover))
            return n, edges
            
    return None, []
