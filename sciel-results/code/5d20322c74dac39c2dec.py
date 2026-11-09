import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 40
    edges_set = set()
    
    # Base: Hamiltonian cycle on vertices 0..39 (the "rim")
    for i in range(N):
        edges_set.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
    
    # Chord strategy: Use chords from vertex 0 to specific vertices.
    # A chord (0, k) creates two cycles:
    # 1. Forward: 0-1-...-k (length k+1)
    # 2. Backward: 0-39-38-...-k (length N-k+1 = 41-k)
    #
    # We need cycles of lengths 3..39. Length 40 is the rim.
    # Let L be the desired cycle length.
    # If we have chord k, we get L1 = k+1 and L2 = 41-k.
    #
    # We want to minimize the number of chords.
    # Let K be the set of chord endpoints (excluding 0 and 1, since (0,1) is a rim edge).
    # Note: Chord (0,1) is already in rim. Chord (0,39) is already in rim.
    # So k can be in {2, ..., 38}.
    #
    # For each k in K, we cover L in {k+1, 41-k}.
    # We need to cover all L in {3, ..., 39}.
    #
    # Let's find a small set K that covers all L.
    # This is a covering problem.
    # Let's try to pick K such that the union of {k+1, 41-k} for k in K is {3..39}.
    #
    # Let's test K = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24}
    # This was found in inspiration 1 to work with 16 chords.
    # Let's verify coverage for this K.
    
    K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24]
    K_set = set(K)
    
    covered = set()
    for k in K:
        covered.add(k + 1)
        covered.add(41 - k)
    
    # Check if all lengths 3..39 are covered
    all_lengths = set(range(3, N))
    missing = all_lengths - covered
    
    # If there are missing lengths, we might need more chords or a different strategy.
    # For K = {2..15, 20, 24}:
    # k=2: 3, 39
    # k=3: 4, 38
    # ...
    # k=15: 16, 26
    # k=20: 21, 21 (duplicate)
    # k=24: 25, 17
    
    # Let's list them:
    # 2 -> 3, 39
    # 3 -> 4, 38
    # 4 -> 5, 37
    # 5 -> 6, 36
    # 6 -> 7, 35
    # 7 -> 8, 34
    # 8 -> 9, 33
    # 9 -> 10, 32
    # 10 -> 11, 31
    # 11 -> 12, 30
    # 12 -> 13, 29
    # 13 -> 14, 28
    # 14 -> 15, 27
    # 15 -> 16, 26
    # 20 -> 21
    # 24 -> 25, 17
    
    # Covered: {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39}
    # Missing: {18, 19, 20, 22, 23, 24}
    
    # The inspiration 1 code claimed this works, but it used "double chords" for the missing ones.
    # Wait, the inspiration 1 code says:
    # "Try double chord: L = 2 + |k1 - k2| => |k1-k2| = L-2"
    # This implies using two chords to form a cycle.
    # Cycle: 0 - u - (u+1) - ... - v - 0
    # This requires chord (0, u) and chord (0, v).
    # The length of this cycle is (v - u) + 2.
    # So L = v - u + 2 => v - u = L - 2.
    # We need to find u, v in K such that v - u = L - 2.
    
    # Let's check if the missing lengths {18, 19, 20, 22, 23, 24} can be formed by pairs in K.
    # K = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24}
    
    # L=18: need diff 16. Pairs in K with diff 16?
    # 20-4=16. Yes. Cycle: 0-4-5-...-20-0. Length 18.
    # L=19: need diff 17.
    # 20-3=17. Yes. Cycle: 0-3-4-...-20-0. Length 19.
    # L=20: need diff 18.
    # 20-2=18. Yes. Cycle: 0-2-3-...-20-0. Length 20.
    # L=22: need diff 20.
    # 24-4=20. Yes. Cycle: 0-4-5-...-24-0. Length 22.
    # L=23: need diff 21.
    # 24-3=21. Yes. Cycle: 0-3-4-...-24-0. Length 23.
    # L=24: need diff 22.
    # 24-2=22. Yes. Cycle: 0-2-3-...-24-0. Length 24.
    
    # All missing lengths are covered by double chords!
    # So 16 chords are sufficient.
    # Total edges = 40 (rim) + 16 (chords) = 56.
    # Score = 56 - 40 = 16.
    
    for k in K:
        edges_set.add((0, k))
    
    # Build cycles
    cycles = {}
    cycles[N] = list(range(N))
    
    for L in range(3, N):
        found = False
        
        # 1. Try single chord (forward)
        k1 = L - 1
        if k1 in K_set:
            cycles[L] = list(range(k1 + 1))
            found = True
        
        # 2. Try single chord (backward)
        if not found:
            k2 = 41 - L
            if k2 in K_set:
                # Cycle: 0, 39, 38, ..., k2
                # Vertices: 0, N-1, N-2, ..., k2
                # Length: 1 + (N-1 - k2 + 1) = N - k2 + 1 = L
                cycles[L] = [0] + list(range(N - 1, k2 - 1, -1))
                found = True
        
        # 3. Try double chord
        if not found:
            diff = L - 2
            for u in K:
                v = u + diff
                if v in K_set and v != u:
                    # Cycle: 0, u, u+1, ..., v
                    # Length: 1 + (v - u) + 1 = v - u + 2 = L
                    cycles[L] = [0] + list(range(u, v + 1))
                    found = True
                    break
        
        if not found:
            # This should not happen with the chosen K
            # Fallback: add necessary chords dynamically?
            # But we want to minimize edges.
            # If we get here, the static K is insufficient.
            # Let's just raise an error or use a fallback.
            # For safety, let's add a chord that covers it, though this increases edge count.
            # But since we verified K works, this block is dead code.
            raise ValueError(f"Cycle length {L} not covered by chord set {K}")
    
    # Validate
    valid = True
    for k in range(3, N + 1):
        if k not in cycles:
            valid = False
            break
        cyc = cycles[k]
        if len(cyc) != k:
            valid = False
            break
        if len(set(cyc)) != k:
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
    
    if valid:
        out = [cycles[k] for k in range(3, N + 1)]
        return {'edges': [list(e) for e in edges_set], 'cycles': out}
    else:
        # Fallback: fan graph (worse score, but valid)
        fan_edges = set()
        for i in range(1, N):
            fan_edges.add((0, i))
        for i in range(1, N - 1):
            fan_edges.add((i, i + 1))
        cycles_fb = []
        for k in range(3, N + 1):
            cycles_fb.append(list(range(k)))
        return {'edges': [list(e) for e in fan_edges], 'cycles': cycles_fb}
