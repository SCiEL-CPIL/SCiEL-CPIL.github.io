N = 40

def construct():
    edges = set()
    
    # Rim: 40 edges
    for i in range(N):
        a, b = i, (i + 1) % N
        if a > b:
            a, b = b, a
        edges.add((a, b))
    
    # Two-hub strategy:
    # Hub 0: chords to vertices 2,3,4,5,6,7,8,9,10,11,12 (11 chords)
    # Hub 20: chords to vertices 22,23,24,25,26,27,28 (7 chords)
    # Total chords: 18... that's more. Let me reconsider.
    
    # Actually, let me try to optimize the single-hub approach more carefully.
    # Current: K = {2,3,...,17,22} = 17 chords
    # Try: K = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,22} = 17 chords
    # or find a 16-chord solution.
    
    # Let me try K = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,22}
    # Singles k+1: 3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,23
    # Singles 41-k: 39,38,37,36,35,34,33,32,31,30,29,28,27,26,22,19
    # Missing from singles: 17,18,21,24,25
    # Doubles:
    # 17: 2+15, |4-19|=15 ✓
    # 18: 2+16, |3-19|=16 ✓  
    # 21: 2+19, |3-22|=19 ✓
    # 24: 2+22, need |a-b|=22. Max is |2-22|=20. ✗
    # 25: 2+23, need |a-b|=23. ✗
    
    # Try K = {2,3,4,5,6,7,8,9,10,11,12,13,14,18,21,22}
    # Singles k+1: 3,4,5,6,7,8,9,10,11,12,13,14,15,19,22,23
    # Singles 41-k: 39,38,37,36,35,34,33,32,31,30,29,28,27,23,20,19
    # Union: {3,4,5,6,7,8,9,10,11,12,13,14,15,19,20,22,23,27,28,29,30,31,32,33,34,35,36,37,38,39}
    # Missing: 16,17,18,21,24,25,26
    # Doubles:
    # 16: 2+14, |4-18|=14 ✓
    # 17: 2+15, |3-18|=15 ✓
    # 18: 2+16, |2-18|=16 ✓
    # 21: 2+19, |2-21|=19 ✓
    # 24: 2+22, need |a-b|=22. Max |2-22|=20. ✗
    
    # It's hard to cover 24,25 with doubles since max |a-b| is limited.
    # Need chords far apart. Let me try including both a small and large chord.
    
    # K = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,22}
    # Singles k+1: 3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,23
    # Singles 41-k: 39,38,37,36,35,34,33,32,31,30,29,28,27,26,21,19
    # Union: {3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,21,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39}
    # Missing: 17,18,20,22,24,25
    # Doubles:
    # 17: 2+15, |5-20|=15 ✓
    # 18: 2+16, |4-20|=16 ✓
    # 20: 2+18, |2-20|=18 ✓
    # 22: 2+20, need |a-b|=20. |2-22|=20 ✓
    # 24: 2+22, need |a-b|=22. Max |2-22|=20. ✗
    # 25: 2+23, ✗
    
    # Still can't get 24,25. Need chords with difference ≥22.
    # Min chord is 2, so need chord ≥24.
    # K = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,24}
    # Singles k+1: 3,4,5,6,7,8,9,10,11,12,13,14,15,16,21,25
    # Singles 41-k: 39,38,37,36,35,34,33,32,31,30,29,28,27,26,21,17
    # Union: {3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,21,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39}
    # Missing: 18,19,20,22,23,24
    # Doubles:
    # 18: 2+16, |4-20|=16 ✓
    # 19: 2+17, |3-20|=17 ✓
    # 20: 2+18, |2-20|=18 ✓
    # 22: 2+20, |4-24|=20 ✓
    # 23: 2+21, |3-24|=21 ✓
    # 24: 2+22, |2-24|=22 ✓
    
    # This works! K = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,24} = 16 chords!
    
    K = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 24]
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
