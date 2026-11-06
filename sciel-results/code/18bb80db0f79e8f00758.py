import random
import time

N = 40

def construct():
    # Required cycle lengths
    required_lengths = set(range(3, N + 1))
    
    # Search for a minimal set S of chords from vertex 0
    # S is a subset of {2, ..., 38}
    # We want to minimize |S|
    
    def get_generated_lengths(S):
        S = list(S)
        lengths = set()
        # Single chords
        for s in S:
            lengths.add(s + 1)
            lengths.add(N + 1 - s)
        # Pair chords
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    # s_i < s_j case handled by abs diff?
                    # Lengths are s_j - s_i + 2 and N + 1 - (s_j - s_i)
                    # This assumes s_j > s_i.
                    # If s_j < s_i, the "path" is the other way.
                    # The formula $d = |s_i - s_j|$ works for the chord pair cycle lengths?
                    # Let's verify.
                    # Cycle 0-s_i-...-s_j-0.
                    # If s_i < s_j, path s_i->s_j has length s_j - s_i. Cycle len s_j - s_i + 2.
                    # If s_i > s_j, path s_i->s_j goes s_i->...->39->0->1->...->s_j?
                    # No, the cycle is 0-s_i-...-s_j-0.
                    # The path between s_i and s_j on the cycle has two parts.
                    # Part 1: s_i -> s_{i+1} -> ... -> s_j (if s_i < s_j) or s_j -> ... -> s_i (if s_j < s_i).
                    # In both cases, the length of the path along the cycle indices is |s_i - s_j|.
                    # So the cycle 0-s_i-path-s_j-0 has length |s_i - s_j| + 2.
                    # The other cycle 0-s_j-path'-s_i-0 has length (N - |s_i - s_j|) + 2?
                    # Path' length is N - |s_i - s_j|.
                    # So lengths are d + 2 and N - d + 2?
                    # Let's re-evaluate N - d + 2.
                    # N=40. d=1. Lengths 3 and 41?
                    # Cycle 0-2-3-0 (len 3). Cycle 0-3-4-...-39-0-1-2-0?
                    # 0-3-...-39-0-1-2-0.
                    # Vertices: 0, 3, 4, ..., 39, 1, 2.
                    # Count: 1 + 37 + 2 = 40?
                    # 3 to 39 is 37 steps. 39 to 0 is 1. 0 to 1 is 1. 1 to 2 is 1. 2 to 0 is 1.
                    # Total edges: 37 + 1 + 1 + 1 + 1 = 41?
                    # Wait.
                    # Cycle 0-s_j-...-39-0-1-...-s_i-0.
                    # Edges: (0, s_j), (s_j, s_j+1)...(39, 0), (0, 1)...(s_i-1, s_i), (s_i, 0).
                    # Count: 1 + (39 - s_j) + 1 + s_i + 1 = 42 - s_j + s_i.
                    # If s_i < s_j, d = s_j - s_i.
                    # 42 - s_j + s_i = 42 - (s_j - s_i) = 42 - d.
                    # So lengths are d + 2 and 42 - d.
                    # For d=1: 3 and 41.
                    # But max cycle length is 40.
                    # So 42-d must be <= 40 => d >= 2.
                    # If d=1, the "other" cycle is the whole graph?
                    # 0-2-3-0 (len 3).
                    # 0-3-4-...-39-0-1-2-0.
                    # This visits 0, 3, 4, ..., 39, 1, 2.
                    # Distinct? Yes.
                    # Length 41?
                    # Vertices: 0, 3, 4, ..., 39, 1, 2.
                    # 0 is counted once.
                    # 3..39 is 37 vertices.
                    # 1, 2 is 2 vertices.
                    # Total 40 vertices.
                    # So length is 40.
                    # Why did I get 41?
                    # Edges: (0,3), (3,4)...(39,0), (0,1), (1,2), (2,0).
                    # 1 + 37 + 1 + 1 + 1 = 41 edges?
                    # (3,4) to (39,0) is 37 edges?
                    # 3->4 (1), ..., 38->39 (36), 39->0 (37). Yes.
                    # So 41 edges?
                    # But a cycle on 40 vertices has 40 edges.
                    # Ah, (0,3) and (2,0) are the chords.
                    # The path 3->...->39->0->1->2 has length 37 + 1 + 1 + 1 = 40?
                    # 3->4 (1) ... 38->39 (36) is 36 edges.
                    # 39->0 (1) is 1 edge.
                    # 0->1 (1) is 1 edge.
                    # 1->2 (1) is 1 edge.
                    # Total path edges: 36 + 1 + 1 + 1 = 39.
                    # Plus (0,3) and (2,0)?
                    # Cycle: 0-3-4-...-39-0-1-2-0.
                    # Edges: (0,3), (3,4), ..., (39,0), (0,1), (1,2), (2,0).
                    # Count: 1 + 36 + 1 + 1 + 1 + 1 = 41?
                    # Wait. 3 to 39 is 36 steps?
                    # 3,4,...,39. Number of vertices = 37.
                    # Number of edges = 36.
                    # So 1 + 36 + 1 + 1 + 1 + 1 = 41.
                    # This is not a simple cycle on 40 vertices?
                    # Vertices: 0, 3, 4, ..., 39, 1, 2.
                    # Set: {0, 1, 2, 3, ..., 39}. All 40.
                    # So it should be length 40.
                    # Where is the extra edge?
                    # (0,3) is chord.
                    # (3,4)...(38,39) is 36 edges.
                    # (39,0) is cycle edge.
                    # (0,1) is cycle edge.
                    # (1,2) is cycle edge.
                    # (2,0) is chord.
                    # Total: 1 + 36 + 1 + 1 + 1 + 1 = 41.
                    # This implies the cycle has 41 edges?
                    # No, a cycle with V vertices has V edges.
                    # V=40.
                    # Let's list them:
                    # 0-3
                    # 3-4
                    # 4-5
                    # ...
                    # 38-39
                    # 39-0
                    # 0-1
                    # 1-2
                    # 2-0
                    # Edges: (0,3), (3,4), ..., (38,39), (39,0), (0,1), (1,2), (2,0).
                    # (3,4) to (38,39) is 36 edges.
                    # Total 1 + 36 + 1 + 1 + 1 + 1 = 41.
                    # This is a contradiction.
                    # Ah, (39,0) and (0,1) share 0.
                    # The sequence is 0, 3, 4, ..., 39, 0, 1, 2, 0.
                    # This is not a simple cycle because 0 appears twice (start and end, and in between?).
                    # 0 -> 3 -> ... -> 39 -> 0 -> 1 -> 2 -> 0.
                    # 0 is visited at step 0 and step 39 (39->0) and step 41 (2->0).
                    # So it's not a simple cycle.
                    # The simple cycle is 0-3-4-...-39-1-2-0?
                    # No, edge (39,1) does not exist.
                    # So for d=1, only length 3 is valid?
                    # And length 40 is the main cycle.
                    # So 42-d is only valid if the path doesn't cross 0.
                    # The path 39->0->1 crosses 0.
                    # So we cannot use that cycle.
                    # Thus, for d=1, we only get length 3.
                    # For d=2 (e.g., 2,4):
                    # 0-2-3-4-0 (len 4).
                    # 0-4-5-...-39-0-1-2-0?
                    # Path 4->...->39->0->1->2.
                    # 4->...->39 is 35 edges.
                    # 39->0 is 1.
                    # 0->1 is 1.
                    # 1->2 is 1.
                    # Total path 38.
                    # Cycle 0-4-...-2-0.
                    # Edges: (0,4), (4,5)...(39,0), (0,1), (1,2), (2,0).
                    # 1 + 35 + 1 + 1 + 1 + 1 = 40.
                    # Vertices: 0, 4, ..., 39, 1, 2.
                    # Distinct? Yes.
                    # So length 40 is valid.
                    # So 42-d is valid if d >= 2.
                    
                    d = abs(S[i] - S[j])
                    if d > 0:
                        lengths.add(d + 2)
                        if d >= 2:
                            lengths.add(42 - d)
                    else:
                        pass
        return lengths

    # Search for S
    best_S = None
    start_time = time.time()
    
    # Try sizes 7 to 12
    for size in range(7, 13):
        if time.time() - start_time > 50:
            break
        found = False
        for _ in range(2000):
            if time.time() - start_time > 50:
                break
            # Sample S
            # Bias towards smaller numbers?
            candidates = list(range(2, 39))
            S = set(random.sample(candidates, size))
            gen = get_generated_lengths(S)
            if required_lengths.issubset(gen):
                best_S = S
                found = True
                break
        if found:
            break
    
    if best_S is None:
        # Fallback to original
        best_S = set(range(2, 21))
        
    # Construct edges
    cycle_edges = [(i, (i + 1) % N) for i in range(N)]
    chord_edges = [(0, k) for k in best_S]
    edges = cycle_edges + chord_edges
    
    # Verify
    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        assert key not in edge_set, f"Duplicate edge {key}"
        edge_set.add(key)
        
    # Construct cycles
    cycles = []
    S_list = sorted(best_S)
    
    for k in range(3, N + 1):
        cyc = None
        # Try single chords
        for s in S_list:
            if k == s + 1:
                cyc = list(range(s + 1))
                break
            elif k == N + 1 - s:
                # 0, s, s+1, ..., 39
                cyc = [0] + list(range(s, N))
                break
        if cyc is None:
            # Try pairs
            for i in range(len(S_list)):
                for j in range(len(S_list)):
                    if i == j: continue
                    s_i = S_list[i]
                    s_j = S_list[j]
                    d = abs(s_i - s_j)
                    if d == 0: continue
                    if k == d + 2:
                        # 0, min, min+1, ..., max
                        a, b = min(s_i, s_j), max(s_i, s_j)
                        cyc = [0] + list(range(a, b + 1))
                        break
                    elif d >= 2 and k == 42 - d:
                        # 0, max, max+1, ..., 39, 0, 1, ..., min
                        a, b = min(s_i, s_j), max(s_i, s_j)
                        cyc = [0] + list(range(b, N)) + list(range(1, a + 1))
                        break
                if cyc is not None:
                    break
        if cyc is None and k == N:
            cyc = list(range(N))
            
        if cyc is None:
            # Should not happen if search worked
            raise ValueError(f"No cycle found for length {k}")
            
        # Verify distinct
        if len(set(cyc)) != k:
            raise ValueError(f"Cycle {k} not distinct")
        # Verify edges
        for idx in range(k):
            u = cyc[idx]
            v = cyc[(idx + 1) % k]
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                raise ValueError(f"Edge {key} missing for cycle {k}")
                
        cycles.append(cyc)
        
    return {'edges': edges, 'cycles': cycles}
