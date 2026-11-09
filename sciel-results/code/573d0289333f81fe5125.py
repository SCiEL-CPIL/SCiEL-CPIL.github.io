import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin

    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    def rim_dist(a, b):
        d = abs(a - b)
        return min(d, N - d)

    def covers_all_lengths(chords):
        """
        Returns True if the set of chords covers all cycle lengths 3..39
        using primary (rim+1 chord) and secondary (rim+2 chords) cycles.
        """
        covered = set()
        # Primary lengths
        for d in chords:
            covered.add(d + 1)
            covered.add(N - d + 1)
        
        # Secondary lengths
        for i in range(len(chords)):
            for j in range(i + 1, len(chords)):
                a = chords[i]
                b = chords[j]
                dist = rim_dist(a, b)
                covered.add(dist + 2)
                covered.add(N - dist + 2)
        
        # Check if 3..39 are all in covered
        for k in range(3, N):
            if k not in covered:
                return False
        return True

    # Initial construction: 19 chords from 0
    # d in 2..20
    initial_chords = list(range(2, 21))
    
    # Local search to minimize chord count
    current_chords = initial_chords[:]
    
    # Try to remove chords one by one
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_chords)):
            # Try removing current_chords[i]
            candidate = current_chords[:i] + current_chords[i+1:]
            if covers_all_lengths(candidate):
                current_chords = candidate
                improved = True
                break # Restart loop with new size
        
    # Build edges
    edges_set = set()
    # Rim
    for i in range(N):
        edges_set.add(norm_edge(i, (i + 1) % N))
        
    # Chords
    for d in current_chords:
        edges_set.add(norm_edge(0, d))
        
    # Build cycles
    # We need to explicitly provide cycles for 3..40.
    # For each length k, we need to find a valid cycle in the graph.
    # We know the graph is pancyclic by construction check.
    # We need to extract the actual vertex lists.
    
    # Helper to find a cycle of length k
    # We can use the knowledge of how the length was covered.
    # But for safety, let's do a simple DFS/BFS or use the structural knowledge.
    
    # Structural knowledge:
    # If k is covered by primary chord d:
    #   Cycle is 0..d or 0..d..39..0
    # If k is covered by secondary chords a, b:
    #   Cycle is 0..a..b..0 (using rim path between a and b)
    
    # We need to map each k to a specific cycle.
    # Let's re-calculate the coverage with details.
    
    cycles = {}
    cycles[N] = list(range(N))
    
    for k in range(3, N):
        found = False
        # Check primary
        for d in current_chords:
            if d + 1 == k:
                # Cycle 0, 1, ..., d
                cycles[k] = list(range(0, d + 1))
                found = True
                break
            if N - d + 1 == k:
                # Cycle 0, d, d+1, ..., 39
                cycles[k] = [0] + list(range(d, N))
                found = True
                break
        if found:
            continue
            
        # Check secondary
        for i in range(len(current_chords)):
            for j in range(i + 1, len(current_chords)):
                a = current_chords[i]
                b = current_chords[j]
                dist = rim_dist(a, b)
                if dist + 2 == k:
                    # Path a to b with rim distance 'dist'
                    # Determine direction
                    if a < b:
                        if b - a == dist:
                            # a -> a+1 -> ... -> b
                            path = list(range(a, b + 1))
                        else:
                            # a -> a-1 -> ... -> 0 -> ... -> b (wrap around)
                            # dist = a + (N - b)
                            path = list(range(a, -1, -1)) + list(range(0, b + 1))
                    else:
                        if a - b == dist:
                            path = list(range(b, a + 1))
                        else:
                            path = list(range(b, -1, -1)) + list(range(0, a + 1))
                    
                    # Cycle: 0 -> a -> path -> b -> 0
                    # Vertices: 0, a, ..., b
                    # Note: 'path' includes a and b.
                    # So cycle is 0, a, path[1:-1], b
                    # Wait, path is the rim segment from a to b.
                    # The cycle is 0 -> a (chord), a -> ... -> b (rim), b -> 0 (chord).
                    # Vertices: 0, a, ..., b.
                    # Ensure distinctness.
                    cyc = [0] + path
                    # Check if valid
                    if len(set(cyc)) == k and len(cyc) == k:
                         # Verify edges
                         valid = True
                         for idx in range(k):
                             u = cyc[idx]
                             v = cyc[(idx + 1) % k]
                             if norm_edge(u, v) not in edges_set:
                                 valid = False
                                 break
                         if valid:
                             cycles[k] = cyc
                             found = True
                             break
                if N - dist + 2 == k:
                    # The other rim path
                    if a < b:
                        if b - a != dist:
                             # This is the long path
                             path = list(range(a, -1, -1)) + list(range(0, b + 1))
                        else:
                             path = list(range(b, -1, -1)) + list(range(0, a + 1))
                    else:
                        if a - b != dist:
                             path = list(range(b, -1, -1)) + list(range(0, a + 1))
                        else:
                             path = list(range(a, -1, -1)) + list(range(0, b + 1))
                    
                    cyc = [0] + path
                    if len(set(cyc)) == k and len(cyc) == k:
                         valid = True
                         for idx in range(k):
                             u = cyc[idx]
                             v = cyc[(idx + 1) % k]
                             if norm_edge(u, v) not in edges_set:
                                 valid = False
                                 break
                         if valid:
                             cycles[k] = cyc
                             found = True
                             break
            if found:
                break
        if not found:
            # Fallback: Should not happen if covers_all_lengths is correct
            # But just in case, use a generic search or raise error
            # For now, return the 19-chord solution if this fails
            pass

    # Verify all cycles found
    valid_result = True
    for k in range(3, N + 1):
        if k not in cycles:
            valid_result = False
            break
        cyc = cycles[k]
        if len(cyc) != k or len(set(cyc)) != k:
            valid_result = False
            break
        for i in range(k):
            if norm_edge(cyc[i], cyc[(i + 1) % k]) not in edges_set:
                valid_result = False
                break
        if not valid_result:
            break

    if not valid_result:
        # Revert to 19 chords
        current_chords = initial_chords
        edges_set = set()
        for i in range(N):
            edges_set.add(norm_edge(i, (i + 1) % N))
        for d in current_chords:
            edges_set.add(norm_edge(0, d))
        
        cycles = {}
        cycles[N] = list(range(N))
        for k in range(3, N):
             # Re-extract for 19 chords
             for d in current_chords:
                 if d + 1 == k:
                     cycles[k] = list(range(0, d + 1))
                     break
                 if N - d + 1 == k:
                     cycles[k] = [0] + list(range(d, N))
                     break
                     
        # Final check
        for k in range(3, N + 1):
            if k not in cycles:
                 # Should not happen
                 pass

    edges_out = [list(e) for e in edges_set]
    cycles_out = [cycles[k] for k in range(3, N + 1)]
    
    return {'edges': edges_out, 'cycles': cycles_out}
