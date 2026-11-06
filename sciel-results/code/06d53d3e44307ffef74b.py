N = 40

def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Backbone cycle edges
    backbone_edges = [(i, (i + 1) % N) for i in range(N)]
    
    # For each possible chord (u,v), determine which cycle lengths it covers
    # A chord (u,v) creates cycles of length d and N-d where d = (v-u) % N
    # We need to cover lengths 3..39 (length 40 is the backbone)
    
    uncovered = set(range(3, N))  # lengths 3 to 39
    
    # Precompute: for each chord, which lengths it covers
    chord_covers = {}
    for u in range(N):
        for v in range(u + 1, N):
            if v == u + 1:
                continue  # backbone edge
            d = v - u
            lengths = set()
            if 3 <= d <= 39:
                lengths.add(d)
            nd = N - d
            if 3 <= nd <= 39:
                lengths.add(nd)
            chord_covers[(u, v)] = lengths
    
    # Greedy set cover with lookahead
    selected_chords = []
    
    while uncovered and (time.time() - start_time) < DEADLINE:
        best_chord = None
        best_gain = 0
        best_chord_lengths = None
        
        for chord, lengths in chord_covers.items():
            gain = len(lengths & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_chord = chord
                best_chord_lengths = lengths
        
        if best_gain == 0:
            break
        
        selected_chords.append(best_chord)
        uncovered -= best_chord_lengths
    
    # Build edge set
    edge_set = set()
    for u, v in backbone_edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    for u, v in selected_chords:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    edges = list(edge_set)
    
    # Generate cycles
    # We need to find actual cycles for each length 3..40
    # Strategy: use the backbone and chords to construct cycles
    
    # Build adjacency for cycle finding
    adj = {i: set() for i in range(N)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    cycles = []
    
    # For length 40: the backbone
    cycles.append(list(range(N)))
    
    # For other lengths, we need to find cycles
    # A chord (u,v) with d = v-u creates a cycle: u, u+1, ..., v, u (length d+1? No...)
    # Wait: chord (u,v) where v = u+d. The cycle going forward is u, u+1, ..., v, then back to u via chord. That's d+1 vertices? No:
    # u to v via backbone: u, u+1, ..., v (that's d edges, d+1 vertices). Then v to u via chord. Total: d+1 vertices? 
    # Actually: vertices are u, u+1, ..., v. That's d+1 vertices. But a cycle of length L has L vertices.
    # So chord (u,v) with d=v-u gives a cycle of length d+1? No wait.
    
    # Let me reconsider. If we have chord (u,v), the cycle is: u, u+1, ..., v, u (back via chord).
    # Number of vertices: u, u+1, ..., v = d+1 vertices. Plus we return to u, but u is already counted.
    # So it's d+1 vertices? No: u to v is d steps, so d+1 vertices in the path. The cycle includes all these vertices.
    # So cycle length = d+1? That doesn't match.
    
    # Actually, I think I was confusing edge count with vertex count. A cycle of length L has L vertices and L edges.
    # If chord connects u and v where v-u = d, then the cycle going through the backbone from u to v has d edges on the backbone, plus 1 chord edge = d+1 edges = d+1 vertices.
    # So the cycle length is d+1.
    
    # Similarly, going the other way: N-d edges on backbone + 1 chord = N-d+1 edges = N-d+1 vertices.
    
    # So chord (u,v) with d = v-u covers cycle lengths d+1 and N-d+1.
    
    # Let me redo: we need to cover lengths 3..39.
    # Chord (u,v) with d=v-u covers lengths d+1 and N-d+1.
    # d ranges from 1 to N-1, but d=1 is backbone, so d from 2 to N-2.
    # d+1 ranges from 3 to N-1 = 39. N-d+1 ranges from 3 to N-1 = 39.
    
    # So each chord covers 2 lengths (or 1 if d+1 = N-d+1, i.e., d = N/2 = 20).
    
    # Let me restart the greedy with correct length computation.
    
    uncovered = set(range(3, N))
    chord_covers2 = {}
    for u in range(N):
        for v in range(u + 2, N):  # skip backbone edges (v=u+1)
            d = v - u
            l1 = d + 1
            l2 = N - d + 1
            lengths = set()
            if 3 <= l1 <= 39:
                lengths.add(l1)
            if 3 <= l2 <= 39:
                lengths.add(l2)
            if lengths:
                chord_covers2[(u, v)] = lengths
    
    selected_chords2 = []
    while uncovered and (time.time() - start_time) < DEADLINE:
        best_chord = None
        best_gain = 0
        best_lengths = None
        
        for chord, lengths in chord_covers2.items():
            gain = len(lengths & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_chord = chord
                best_lengths = lengths
        
        if best_gain == 0:
            break
        
        selected_chords2.append(best_chord)
        uncovered -= best_lengths
    
    # Rebuild edges with correct chords
    edge_set = set()
    for u, v in backbone_edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    for u, v in selected_chords2:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    edges = list(edge_set)
    
    # Now generate cycles. For each length L in 3..40:
    # If L == 40: backbone
    # If L < 40: find a chord that covers L, then construct the cycle
    
    # Map: for each chord, what cycles it creates
    chord_to_cycles = {}
    for (u, v) in selected_chords2:
        d = v - u
        l1 = d + 1
        l2 = N - d + 1
        # Cycle 1: u, u+1, ..., v (length l1)
        if 3 <= l1 <= 39:
            cyc1 = list(range(u, v + 1))
            chord_to_cycles[(u, v, l1)] = cyc1
        # Cycle 2: v, v+1, ..., 39, 0, 1, ..., u (length l2)
        if 3 <= l2 <= 39:
            cyc2 = list(range(v, N)) + list(range(0, u + 1))
            chord_to_cycles[(u, v, l2)] = cyc2
    
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
        else:
            # Find a chord that covers length L
            found = False
            for (u, v) in selected_chords2:
                d = v - u
                l1 = d + 1
                l2 = N - d + 1
                if l1 == L:
                    cycles.append(list(range(u, v + 1)))
                    found = True
                    break
                elif l2 == L:
                    cycles.append(list(range(v, N)) + list(range(0, u + 1)))
                    found = True
                    break
            if not found:
                # Fallback: try to find any cycle of length L
                # This should not happen if set cover worked
                cycles.append(list(range(L)))
    
    return {'edges': edges, 'cycles': cycles}
