import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0
    N = 40

    # Two-hub decomposition: hubs at 0 and 20
    # Rim edges
    edges_set = set()
    for i in range(N):
        edges_set.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))

    # Hub 0: chords to cover lengths 3-20
    # Chord (0,d) for d=2..19 gives forward length d+1 (3-20)
    for d in range(2, 20):
        edges_set.add((0, d))

    # Hub 20: chords to cover lengths 21-39
    # Chord (20,d) for d=21..38 gives:
    #   Forward in segment: 20,21,...,d has length d-20+1 = d-19 (2..19)
    #   Backward: 20,19,18,...,d-1 has length 40-(d-20)+1 = 61-d (23..40)
    # We need lengths 21-39. Use backward cycles from hub 20.
    # Chord (20,d): backward cycle 20,19,...,d-1 has length (20-(d-1))+1 = 22-d+1... 
    # Let me recalculate: path from 20 going backward (decreasing) to d-1
    # 20, 19, 18, ..., d-1: that's 20-(d-1)+1 = 22-d vertices? No.
    # From 20 down to d-1: vertices are 20,19,...,d-1. Count = 20-(d-1)+1 = 22-d.
    # Plus the chord closes it. So cycle length = 22-d + 1? No, the chord is the closing edge.
    # Cycle: 20, 19, ..., d-1, 20 (via chord). Length = (20 - (d-1) + 1) = 22-d.
    # For d=21: length = 22-21 = 1? That's wrong.
    
    # Let me rethink. From hub 20, chord to vertex d.
    # Forward path: 20, 21, 22, ..., d. Length = d - 20 + 1 = d-19.
    # Backward path: 20, 19, 18, ..., d-1 (wrapping around). 
    #   If d > 20: 20, 19, ..., 0, 39, 38, ..., d-1. Length = 20 + (40-(d-1)) = 20 + 41-d = 61-d.
    #   Wait: from 20 going backward: 20,19,...,0 (21 vertices), then 39,38,...,d-1 (40-(d-1) = 41-d vertices).
    #   Total = 21 + 41-d = 62-d. Hmm, that doesn't seem right either.
    #   Actually: 20,19,...,0 is 21 vertices. Then 39,38,...,d-1 is 39-(d-1)+1 = 41-d vertices.
    #   But we don't double count. The path is 20->19->...->0->39->...->d-1->20(chord).
    #   Number of vertices = 21 + (41-d) = 62-d. But that's the number of vertices in the path,
    #   and the cycle has the same number of vertices.
    #   For d=21: 62-21 = 41. That's too many (max 40).
    
    # I think I'm overcomplicating this. Let me just use the same approach as before
    # but split the work between two hubs to see if we can reduce total chords.
    
    # Actually, the current single-hub approach already achieves the theoretical minimum
    # of 19 chords for a cycle-based construction. Let me try a completely different
    # structure: a complete bipartite-like or grid structure.
    
    # Alternative: Use a "crown" structure. 
    # Keep the 40-cycle rim. Add chords strategically.
    # The minimum number of chords needed: we need 38 cycle lengths.
    # Each chord can help form at most 2 fundamental cycles with the rim.
    # So we need at least ceil(38/2) = 19 chords. The current solution achieves this.
    
    # To do better, we need chords that participate in more than 2 cycles.
    # This requires a denser structure. Let me try adding a second rim.
    
    # New approach: Two concentric 20-cycles with rungs.
    # Outer: 0-1-...-19-0 (20 edges)
    # Inner: 20-21-...-39-20 (20 edges)  
    # Rungs: (0,20), (1,21), ..., (19,39) (20 edges)
    # Total so far: 60 edges. That's worse.
    
    # Let me try: one 40-cycle rim (40 edges) + 18 chords instead of 19.
    # Is it possible with 18 chords? We'd need some chord to cover 3+ cycle lengths.
    
    # Try: rim + chords (0,2),(0,3),...,(0,19) [18 chords] = 58 edges
    # This covers: forward 3-20, backward 22-39. Missing: 21.
    # Can we get 21 from existing edges? 
    # Cycle of length 21: need 21 distinct vertices forming a cycle.
    # With chords from 0 to 2-19, we can form: 0,1,2,...,k (length k+1) or 0,k,k+1,...,39 (length 41-k)
    # For length 21: k+1=21 -> k=20 (no chord to 20), or 41-k=21 -> k=20 (same).
    # So we can't get 21 without a chord to 20.
    
    # What if we use a chord that's not from vertex 0?
    # E.g., chord (1,21). This gives:
    #   1,2,...,21 (length 21) and 1,0,39,38,...,22 (length 40-21+2=21? Let me check)
    #   Forward: 1,2,...,21 = 21 vertices. Cycle length 21.
    #   Backward: 1,0,39,38,...,22 = 1+1+18 = 20 vertices? 1,0 (2), 39 down to 22 (18) = 20. 
    #   Wait: 39,38,...,22 is 39-22+1=18 vertices. Plus 1 and 0 = 20. Not 21.
    #   The cycle is 1-0-39-...-22-1. Edges: (1,0) rim, (0,39) rim, ..., (22,1) chord.
    #   Vertices: 1,0,39,38,...,22. Count = 2 + 18 = 20. So length 20, not 21.
    
    # Let me just keep the current 59-edge solution which is optimal for this structure.
    # The cost is 19, which gives score 37/19 ≈ 1.947.
    
    # Revert to the known-good 59-edge solution
    edges_set = set()
    for i in range(N):
        edges_set.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))

    for d in range(2, 19):
        edges_set.add((0, d))
    edges_set.add((2, 21))

    cycles = {}
    cycles[40] = list(range(N))

    for d in range(2, 19):
        fwd = list(range(0, d + 1))
        bwd = [0] + list(range(d, N))
        if 3 <= len(fwd) <= 40 and len(fwd) not in cycles:
            cycles[len(fwd)] = fwd
        if 3 <= len(bwd) <= 40 and len(bwd) not in cycles:
            cycles[len(bwd)] = bwd

    cycles[20] = list(range(2, 22))
    cycles[21] = [2, 21] + list(range(22, N)) + [0, 1]
    cycles[22] = [0, 1, 2, 21] + list(range(22, N))

    valid = True
    for k in range(3, N + 1):
        if k not in cycles:
            valid = False
            break
        cyc = cycles[k]
        if len(cyc) != k or len(set(cyc)) != k:
            valid = False
            break
        for i in range(k):
            a = cyc[i]
            b = cyc[(i + 1) % k]
            if (min(a, b), max(a, b)) not in edges_set:
                valid = False
                break
        if not valid:
            break

    if not valid or time.time() > DEADLINE:
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = [list(range(k)) for k in range(3, N + 1)]
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}

    out = [cycles[k] for k in range(3, N + 1)]
    return {'edges': [list(e) for e in edges_set], 'cycles': out}
