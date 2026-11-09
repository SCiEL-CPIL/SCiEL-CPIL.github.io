N = 40

def _find_15_chord_solution():
    """Search for a 15-chord set K that covers all cycle lengths 3-39."""
    import itertools
    import time
    
    start = time.time()
    deadline = start + 80.0  # 80s budget, 10s safety margin
    
    # For a chord set K, cycle length L is covered if:
    # - L-1 in K (single chord: 0-1-2-...-(L-1)-0)
    # - 41-L in K (single chord: 0-(N-1)-...-(41-L)-0)  
    # - There exist a,b in K with |a-b| = L-2 (double chord)
    
    # We need to cover L = 3..39
    # L=3: need 2 in K or 38 in K or |a-b|=1 (impossible since min chord is 2)
    #   So 2 must be in K (or 38, but 38>20 so not useful for doubles)
    # Actually 38 in K gives cycle 0-39-38-0 (length 3) ✓
    
    # Let's think about what's needed:
    # Required: for each L in 3..39, at least one of {L-1, 41-L} in K, or some |a-b|=L-2 for a,b in K
    # 
    # Key insight: if 2 is in K, then for any other k in K, we get cycle length 2+|k-2|=k (if k>2) 
    # or 2+(2-k) (if k<2, impossible). So 2+k gives length k for k in K, k>=3.
    # Wait: |a-b| where a=2, b=k gives |k-2|. Cycle length = 2 + |k-2|.
    # If k > 2: length = 2 + k - 2 = k
    # So having 2 in K, every other chord k gives a cycle of length k.
    # 
    # Also single chords: k in K gives cycle length k+1 and 41-k.
    # 
    # So with 2 in K and set K:
    # Covered lengths: {k for k in K, k>=3} (from doubles with 2) 
    #                 ∪ {k+1 for k in K} (from singles)
    #                 ∪ {41-k for k in K} (from singles)
    #                 ∪ {2+|a-b| for a,b in K} (from all doubles)
    
    # Let's search. We need |K|=15, all in {2,...,39}, and 2 in K (likely).
    # Actually 2 doesn't have to be in K. Let me be more general.
    
    # For efficiency, note that large chords (>20) only help via 41-k (giving small lengths) 
    # and via doubles with other large chords.
    # A chord k>20: 41-k < 21, so it covers length 41-k via single.
    # And k+1 > 21, covers length k+1 via single.
    
    # Let's try: assume 2 in K. Then K = {2} ∪ S where |S|=14, S ⊆ {3,...,39}.
    # Covered: {s for s in S} ∪ {s+1 for s in S} ∪ {41-s for s in S} ∪ {2+|a-b| for a,b in S∪{2}}
    # The last set includes {2+|s-2| = s for s in S} (already covered) and {2+|a-b| for a,b in S}
    
    # So effectively covered = {s, s+1, 41-s for s in S} ∪ {2+|a-b| for a,b in S, a≠b}
    # Need this to contain {3,...,39}.
    
    # This is still a large search space. Let's use a greedy/backtracking approach.
    
    # Required lengths: 3..39 (37 lengths)
    # Each chord s covers at most: s, s+1, 41-s (3 singles) plus contributions to doubles.
    
    # Let me try a different approach: start from the known 16-chord solution and try to remove one.
    # K16 = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,24}
    # Try removing each element and check if still covered.
    
    K16 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24]
    
    def covers_all(K):
        K_set = set(K)
        for L in range(3, 40):
            # Single: L-1 in K
            if (L - 1) in K_set:
                continue
            # Single: 41-L in K
            if (41 - L) in K_set:
                continue
            # Double: need a,b in K with |a-b| = L-2
            diff = L - 2
            found = False
            for a in K:
                if (a + diff) in K_set:
                    found = True
                    break
                if (a - diff) in K_set:
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Check that K16 works
    assert covers_all(K16), "K16 should work"
    
    # Try removing one element
    for i in range(len(K16)):
        K_try = K16[:i] + K16[i+1:]
        if covers_all(K_try):
            return K_try
    
    # If no single removal works, try removing one and adding another
    if time.time() > deadline:
        return K16  # fallback to 16
    
    # Try: remove one, add one from {16,17,18,19,21,22,23,25,...,39}
    candidates_to_add = [16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    
    for i in range(len(K16)):
        removed = K16[i]
        K_base = K16[:i] + K16[i+1:]
        for add in candidates_to_add:
            if add == removed or add in K16:
                continue
            K_try = sorted(K_base + [add])
            if covers_all(K_try):
                return K_try
        if time.time() > deadline:
            break
    
    # Try: remove two, add two
    if time.time() > deadline:
        return K16
    
    for i in range(len(K16)):
        for j in range(i+1, len(K16)):
            K_base = K16[:i] + K16[i+1:j] + K16[j+1:]
            for add1 in candidates_to_add:
                if add1 in K16:
                    continue
                for add2 in candidates_to_add:
                    if add2 in K16 or add2 <= add1:
                        continue
                    K_try = sorted(K_base + [add1, add2])
                    if covers_all(K_try):
                        return K_try
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return K16  # fallback


def construct():
    edges = set()
    
    # Rim: 40 edges
    for i in range(N):
        a, b = i, (i + 1) % N
        if a > b:
            a, b = b, a
        edges.add((a, b))
    
    # Find optimal chord set
    K = _find_15_chord_solution()
    
    for k in K:
        a, b = 0, k
        if a > b:
            a, b = b, a
        edges.add((a, b))
    
    edges = sorted(edges)
    
    # Build cycle certificates
    K_set = set(K)
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
                    cyc = [0] + list(range(u, v1 + 1))
                    cycles.append(cyc)
                    found = True
                    break
                elif v2 in K_set and v2 != u:
                    cyc = [0] + list(range(v2, u + 1))
                    cycles.append(cyc)
                    found = True
                    break
        
        if not found:
            raise ValueError(f"Cycle {length} not found")
    
    return {'edges': edges, 'cycles': cycles}
