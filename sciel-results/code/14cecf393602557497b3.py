import time
from itertools import combinations

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute base cycle edges (the 40-cycle)
    base_edges = [(i, (i + 1) % N) for i in range(N)]

    required = set(range(3, N + 1))
    
    # Precompute for each chord s in 2..38:
    #   single_lengths: lengths covered by chord s alone
    #   pair_lengths[s1][s2]: length covered by pair (s1, s2)
    chord_singles = {}
    for s in range(2, N - 1):
        lengths = set()
        l1 = s + 1
        if 3 <= l1 <= N:
            lengths.add(l1)
        l2 = N - s + 1
        if 3 <= l2 <= N:
            lengths.add(l2)
        chord_singles[s] = lengths
    
    # Pair lengths: for a < b, the two chords (0,a) and (0,b) create a cycle
    # 0-a-(a+1)-...-b-0 of length b - a + 2
    chord_pairs = {}
    for a in range(2, N - 1):
        for b in range(a + 1, N - 1):
            l = b - a + 2
            if 3 <= l <= N:
                if a not in chord_pairs:
                    chord_pairs[a] = set()
                if b not in chord_pairs:
                    chord_pairs[b] = set()
                chord_pairs[a].add(l)
                chord_pairs[b].add(l)
    
    def compute_covered(S):
        """Compute covered lengths given chord set S."""
        cov = {N}
        S_list = sorted(S)
        for s in S_list:
            cov.update(chord_singles[s])
        for i in range(len(S_list)):
            a = S_list[i]
            if a not in chord_pairs:
                continue
            cp_a = chord_pairs[a]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                l = b - a + 2
                if 3 <= l <= N and l in cp_a:
                    cov.add(l)
        return cov
    
    def incremental_cover(S, add_s):
        """Compute newly covered lengths when adding chord add_s to set S."""
        new_lengths = set()
        S_list = sorted(S)
        # Single contributions from add_s
        new_lengths.update(chord_singles[add_s])
        # Pair contributions with existing chords
        if add_s in chord_pairs:
            cp = chord_pairs[add_s]
            for s in S_list:
                if s == add_s:
                    continue
                a, b = (s, add_s) if s < add_s else (add_s, s)
                l = b - a + 2
                if 3 <= l <= N and l in cp:
                    new_lengths.add(l)
        return new_lengths

    candidates = list(range(2, N - 1))  # 2..38
    
    best_S = None
    best_size = float('inf')
    
    import random
    rng = random.Random(42)
    
    # Multiple greedy restarts
    for trial in range(500):
        if time.time() > deadline - 10.0:
            break
        
        S = set()
        covered = {N}
        
        while not required.issubset(covered):
            best_gain = -1
            best_s = None
            for s in candidates:
                if s in S:
                    continue
                new_cov = incremental_cover(S, s)
                gain = len(new_cov & required - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            if best_s is None or best_gain <= 0:
                break
            S.add(best_s)
            covered |= incremental_cover(S - {best_s}, best_s)
        
        if required.issubset(covered):
            if len(S) < best_size:
                best_size = len(S)
                best_S = S.copy()
                if best_size <= 8:
                    break
    
    if best_S is None:
        # Fallback: use a known good construction
        best_S = set(range(2, 39, 3))
    
    # Local search: try removals and swaps
    if time.time() < deadline - 15.0:
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            S_list = sorted(best_S)
            
            # Try removals
            for i in range(len(S_list)):
                if time.time() > deadline - 15.0:
                    break
                trial_S = best_S - {S_list[i]}
                if required.issubset(compute_covered(trial_S)):
                    best_S = trial_S
                    improved = True
                    break
            
            # Try swaps: remove one, add another
            if not improved and len(S_list) >= 2:
                for i in range(len(S_list)):
                    if time.time() > deadline - 15.0:
                        break
                    for s in candidates:
                        if s in best_S:
                            continue
                        trial_S = (best_S - {S_list[i]}) | {s}
                        if len(trial_S) == len(best_S) and required.issubset(compute_covered(trial_S)):
                            best_S = trial_S
                            improved = True
                            break
                    if improved:
                        break
    
    S = best_S
    S_sorted = sorted(S)
    
    # Build the edge list
    edges = list(base_edges)
    for s in S:
        edges.append((0, s))
    
    # Remove duplicate edges
    unique_edges = []
    seen = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            unique_edges.append((u, v))
    edges = unique_edges
    
    # Build cycles for each length 3..N
    cycles_by_len = {}
    cycles_by_len[N] = list(range(N))
    
    # From single chords: chord (0,s) gives cycles of length s+1 and N-s+1
    for s in S_sorted:
        l1 = s + 1
        if 3 <= l1 <= N and l1 not in cycles_by_len:
            # Cycle: 0-1-2-...-s-0 (length s+1)
            cycles_by_len[l1] = list(range(s + 1))
        
        l2 = N - s + 1
        if 3 <= l2 <= N and l2 not in cycles_by_len:
            # Cycle: 0-(s-1)-(s-2)-...-1-...-N-1-0? 
            # Chord (0,s) creates cycle 0-s-(s-1)-...-1-0 (length s) or 0-s-(s+1)-...-N-1-0 (length N-s+1)
            # Wait: 0-s is a chord. Path from s to 0 not using chord: s-(s+1)-...-N-1-0 (length N-s)
            # So cycle: 0-s-(s+1)-...-N-1-0, length N-s+1 ✓
            cycles_by_len[l2] = [0] + list(range(s, N))
    
    # From pairs of chords (0,a) and (0,b), a < b:
    # Type 1: 4-cycle 0-a-b-0, length = b - a + 2
    # Type 2: 4-cycle a-0-b-(b+1)-a, length = N - (b - a) + 2
    for i in range(len(S_sorted)):
        for j in range(i + 1, len(S_sorted)):
            a = S_sorted[i]
            b = S_sorted[j]
            d = b - a
            l1 = d + 2
            if 3 <= l1 <= N and l1 not in cycles_by_len:
                # Cycle: 0-a-(a+1)-...-b-0? No.
                # 0-a, a-(a+1)...(b-1)-b, b-0: that's length b-a+2
                # Vertices: 0, a, a+1, ..., b, back to 0
                if a > 0:
                    cyc = [0] + list(range(a, b + 1))
                    cycles_by_len[l1] = cyc
                else:
                    cycles_by_len[l1] = list(range(b + 1))
            
            l2 = N - d + 2
            if 3 <= l2 <= N and l2 not in cycles_by_len:
                # Cycle: a-0-b-(b+1)-...-(a-1 mod N)-a
                # Vertices: a, 0, b, b+1, ..., N-1, 1, 2, ..., a-1 (if a>0)
                # Length: 1 + 1 + (N - b) + (a - 1) = N - b + a + 1 = N - (b-a) + 1... 
                # Wait: a, 0, b, b+1, ..., N-1, 1, 2, ..., a-1
                # Count: a, 0 (2), b to N-1 (N-b), 1 to a-1 (a-1)
                # Total: 2 + (N-b) + (a-1) = N - b + a + 1 = N - d + 1
                # That's N-d+1, not N-d+2. Let me reconsider.
                # Actually the cycle is: a - 0 - b - (b+1) - ... - (N-1) - 1 - 2 - ... - (a-1) - a
                # Edges: (a,0), (0,b), (b,b+1), ..., (N-1,0)? No, (N-1) connects to 0?
                # The base cycle has edge (N-1, 0). So:
                # a-0 (chord), 0-b (chord), b-(b+1) (base), ..., (N-1)-0 (base)? 
                # No, that would close too early.
                # Let me think again. We need a 4+ cycle using both chords.
                # Path from a to b NOT through 0: a-(a+1)-...-b (length b-a edges)
                # Path from a to b through 0: a-0-b (length 2 edges)
                # Combined cycle: a-0-b-(b+1)-...-a? No, b to a going forward wraps.
                # a-0-b-(b+1)-...-(N-1)-0? No, 0 already used.
                # 
                # Alternative: use the OTHER path between a and b on the cycle.
                # Path 1: a-(a-1)-...-0-1-...-b (going the long way around through 0)
                #   Length: a + b edges (from a down to 0, then up to b)
                #   Wait: a to 0 is a edges (a, a-1, ..., 1, 0), 0 to b is b edges
                #   Total path length: a + b
                # Path 2: a-(a+1)-...-b (going forward)
                #   Length: b - a edges
                # 
                # Cycle using chord (0,a) and chord (0,b):
                # We need a cycle that uses both chords. 
                # Option: 0-a-(a+1)-...-b-0 (uses both chords and path a to b forward)
                #   Length: 2 + (b-a) = b-a+2 ✓ (this is l1)
                # Option: 0-b-(b+1)-...-(N-1)-0? No, that doesn't use chord (0,a).
                # 
                # Another option: a-0-b-(b+1)-...-(N-1)-...-a
                # From b forward to N-1, then N-1 to 0? No, 0 is used.
                # 
                # Hmm, let me reconsider. With chords (0,a) and (0,b):
                # The graph has the base cycle plus these two chords.
                # Cycles must use at least one chord.
                # 
                # Cycles using both chords:
                # 0-a-(a+1)-...-b-0: length b-a+2
                # 0-b-(b-1)-...-a-0: length b-a+2 (same length, different direction)
                # 
                # Are there other cycles using both chords?
                # 0-a-(a-1)-...-1-...-b-0? That goes from a backward to 1, then forward to b.
                # But 1 to b goes through 0? No, 1-2-...-b doesn't pass through 0.
                # Wait, the base cycle is 0-1-2-...-N-1-0.
                # Path from a to b not through 0, going backward: a-(a-1)-...-1-2-...-b? No,
                # that passes through 0 if we go a to 0 to b.
                # 
                # Actually on a cycle graph, there are exactly two paths between any two vertices.
                # Between a and b (a < b):
                #   Path 1 (forward): a-(a+1)-...-b, length b-a
                #   Path 2 (backward): a-(a-1)-...-0-...-b, length N-(b-a)
                # 
                # Cycles using both chords (0,a) and (0,b):
                #   Using Path 1: 0-a-[Path1]-b-0, length 2 + (b-a) = b-a+2
                #   Using Path 2: 0-a-[Path2]-b-0, length 2 + (N-(b-a)) = N-(b-a)+2
                # 
                # So yes, l2 = N - d + 2 is valid!
                # Cycle: 0-a-(a-1)-...-1-...-b-0
                # Vertices: 0, a, a-1, a-2, ..., 1, 2, ..., b
                # Wait, that's not right either. Path 2 goes a to b the long way.
                # a-(a-1)-...-1-0? No, 0 is an endpoint.
                # 
                # Path 2 from a to b: a, a-1, a-2, ..., 1, 0, N-1, N-2, ..., b+1, b
                # But this passes through 0! And 0 is already in our cycle.
                # 
                # I think the issue is that in the cycle graph, the two paths between a and b
                # are:
                #   Forward: a, a+1, ..., b (length b-a)
                #   Backward: a, a-1, ..., 0, N-1, ..., b+1, b (length N-(b-a))
                # 
                # The backward path passes through 0. So a cycle using both chords and the
                # backward path would be: 0-a-(a-1)-...-1-0? No, that's a sub-cycle.
                # 
                # Actually, I think l2 = N-d+2 corresponds to:
                # 0-b-(b+1)-...-(N-1)-0? No, that's just the base cycle segment.
                # 
                # Let me just verify with a small example. N=6, a=1, b=3.
                # d=2, l1=4, l2=6-2+2=6.
                # l1=4: 0-1-2-3-0. Edges: (0,1),(1,2),(2,3),(3,0). ✓
                # l2=6: 0-1-5-4-3-2-0? That's length 6 but uses edge (0,1) and (3,0)?
                # Wait, 0-1-5-4-3-2-0: edges (0,1),(1,5),(5,4),(4,3),(3,2),(2,0).
                # (1,5) is a base edge? Base cycle: 0-1-2-3-4-5-0. So (1,5) is NOT a base edge.
                # 
                # I think my l2 formula is wrong. Let me reconsider.
                # 
                # With chords (0,a) and (0,b), the new cycles are:
                # 1. 0-a-[forward path a to b]-b-0: length (b-a)+2
                # 2. 0-b-[forward path b to a wrapping]-a-0: length (N-(b-a))+2? 
                #    Forward path from b to a: b, b+1, ..., N-1, 0, 1, ..., a
                #    But this passes through 0!
                # 
                # I think there's only ONE new cycle length from a pair of chords in a fan:
                # b-a+2. The "other" length N-(b-a)+2 would require a path that doesn't
                # pass through 0, but the only such path is the forward one.
                # 
                # So I should REVERT the full-pair model and keep only l1 = b-a+2.
                # Let me fix this.
                pass
    
    # Check if all lengths are covered
    for L in range(3, N + 1):
        if L not in cycles_by_len:
            # Fallback
            if L <= 21:
                cycles_by_len[L] = list(range(L))
            elif L <= 39:
                k = 41 - L
                cycles_by_len[L] = [0] + list(range(39, k - 1, -1))
            else:
                cycles_by_len[L] = list(range(N))
    
    final_cycles = [cycles_by_len[L] for L in range(3, N + 1)]
    
    # Verify all cycles are valid (edges exist)
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    for L, cyc in zip(range(3, N + 1), final_cycles):
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set:
                # Fallback to a safe construction
                return {
                    'edges': base_edges + [(0, k) for k in range(2, 21)],
                    'cycles': [
                        list(range(L)) if L <= 21 else ([0] + list(range(39, 41 - L - 1, -1)) if L <= 39 else list(range(N)))
                        for L in range(3, N + 1)
                    ]
                }
    
    return {'edges': edges, 'cycles': final_cycles}
