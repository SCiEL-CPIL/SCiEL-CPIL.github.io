"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    # Two-hub decomposition: hubs at 0 and 20
    # Base structure:
    #   - Path through left side: 0-1-2-...-19 (19 edges)
    #   - Path through right side: 20-21-22-...-39 (19 edges)
    #   - Edge between hubs: (0, 20) (1 edge)
    # Total base: 39 edges
    #
    # We need cycles of every length 3..40.
    #
    # Strategy: Use chords to create cycles of various lengths.
    # The key insight is that with two hubs, we can form cycles that
    # go from hub 0, through some left vertices, back to 0, or
    # from hub 0 through hub 20 and some right vertices, etc.
    #
    # Let's design a specific construction:
    #
    # Left side vertices: 1, 2, ..., 19 (connected in a path from 0)
    # Right side vertices: 21, 22, ..., 39 (connected in a path from 20)
    #
    # Base edges:
    #   Left path: (0,1), (1,2), ..., (18,19)  -> 19 edges
    #   Right path: (20,21), (21,22), ..., (38,39) -> 19 edges
    #   Hub connection: (0, 20) -> 1 edge
    # Total: 39 edges
    #
    # Now we need to add chords to enable cycles of all lengths 3..40.
    #
    # A cycle of length k must use k distinct vertices and k edges.
    #
    # Let's think about what cycles we can form:
    #
    # Cycle of length 3: 0-1-20-0? No, (1,20) doesn't exist.
    # We need triangle: 0-1-? -0. If we add chord (1,20), then 0-1-20-0 is a triangle.
    # But we also need (0,20) which we have. So triangle: 0,1,20 with edges (0,1),(1,20),(0,20).
    #
    # Let's try a different approach. Use a "ladder" or "fan" structure.
    #
    # Actually, let me think more carefully. The known result mentions:
    # "Two-hub decomposition with hubs at 0 and 20 allows covering cycle lengths 3-39 
    #  with 16 total chords instead of 17, reducing edge count from 57 to 56"
    #
    # So the target is 56 edges total, meaning 56-40 = 16 excess edges.
    # Wait, the score is "edge count minus 40". So if we have 56 edges, score = 16.
    # The seed has: 39 (path edges) + 1 (hub) = 40 edges... wait let me recount.
    #
    # Seed: [(0,i) for i in range(1,40)] = 39 edges (star from 0)
    #      + [(i,i+1) for i in range(1,39)] = 38 edges (path 1-2-...-39)
    # Total seed edges = 39 + 38 = 77. Score = 77-40 = 37.
    #
    # Wait, that can't be right for score 1.0. Let me re-read.
    # The seed returns edges: star from 0 to all others (39 edges) + path 1-2-...-39 (38 edges).
    # But (0,1) is in both! So it's a set, duplicates removed? No, they're listed as pairs.
    # Actually looking again: [(0,i) for i in range(1,N)] gives (0,1)...(0,39) = 39 edges
    # [(i,i+1) for i in range(1,N-1)] gives (1,2)...(38,39) = 38 edges
    # Total = 77 edges. Score = 77 - 40 = 37.
    #
    # But the score is reported as 1.0. So the metric must be different.
    # "Minimize edge count minus 40" - maybe it's (edge_count - 40) and lower is better,
    # and the search score is seed/value where value is the metric.
    # If seed metric = 37 and score = 1.0, then seed_value = 37.
    # A child with metric = 16 would have score = 37/16 ≈ 2.3125. That matches "2.176 -> 2.312"!
    #
    # So target: minimize (num_edges - 40). Best known: 56 edges -> metric 16.
    #
    # Let me construct a 56-edge graph.
    #
    # Base: 39 edges (two paths + hub edge)
    # Need 56 - 39 = 17 more edges (chords).
    #
    # Wait, the note says 16 chords reducing from 57 to 56. So maybe base is 40 edges?
    # Let me just aim for 56 edges total.
    #
    # Construction:
    # Vertices 0-39.
    # Left chain: 0-1-2-...-19 (19 edges)
    # Right chain: 20-21-22-...-39 (19 edges)  
    # Hub edge: (0,20) (1 edge)
    # Total base: 39 edges.
    #
    # For cycles, I'll use a systematic chord placement.
    # 
    # Key idea: To get a cycle of length k, I can use:
    #   - A path from 0 to some vertex v on the left (length d1)
    #   - A chord from v to some vertex w on the right (or to 20)
    #   - A path from w to 20 (length d2)
    #   - The hub edge (0,20)
    #   Cycle length = d1 + 1 + d2 + 1 = d1 + d2 + 2
    #
    # Alternatively, use chords within one side.
    #
    # Let me try a concrete construction with 56 edges:
    #
    # Edges:
    # 1. Left path: (0,1),(1,2),...,(18,19) = 19 edges
    # 2. Right path: (20,21),(21,22),...,(38,39) = 19 edges
    # 3. Hub: (0,20) = 1 edge
    # 4. Chords: 17 chords to be determined
    #
    # Total: 39 + 17 = 56 edges.
    #
    # Let me design chords to cover all cycle lengths 3-40.
    #
    # I'll use the following chord strategy:
    # For each "gap" in cycle length, add a chord that enables it.
    #
    # Actually, let me use a known good construction pattern:
    # Add chords (i, 20+i) for appropriate i, and some within-side chords.
    #
    # Let me just hardcode a working 56-edge construction.
    #
    # Chords (17 total):
    # c1: (1,20)  -> enables triangle 0-1-20-0 (len 3)
    # c2: (2,20)  -> enables 0-1-2-20-0 (len 4)
    # c3: (3,20)  -> enables 0-1-2-3-20-0 (len 5)
    # ...
    # ci: (i,20) for i=1..17 -> enables cycles of length i+2 (3..19)
    # That's 17 chords giving cycles 3-19.
    #
    # But I also need cycles 20-40.
    #
    # For longer cycles, I can use the right side path.
    # Cycle using left path to vertex i, chord to 20, then right path.
    # 0-1-...-i-20-21-...-20+j-... back to 0? No, need to close.
    #
    # Alternative: use chords from left vertices to right vertices (not just to 20).
    #
    # Let me reconsider. With hub edge (0,20):
    # A cycle can go: 0 -> (left path) -> v -> (chord) -> w -> (right path) -> 20 -> (hub) -> 0
    # Length = dist_left(0,v) + 1 + dist_right(20,w) + 1
    #
    # If chord is (v,w) where v is on left (1..19) and w is on right (21..39):
    # dist_left(0,v) = v (if v is at position v on path 0-1-...-19)
    # dist_right(20,w) = w-20
    # Cycle length = v + 1 + (w-20) + 1 = v + w - 18
    #
    # Also cycles entirely on one side with a chord back:
    # Left side: 0-1-...-v-(chord to 0)-0: length = v + 1. Need chord (v,0) but (0,v) is already in path for v<=19. So that's just the path, not a cycle unless we have a chord.
    # Actually 0-1-...-v-0 requires edge (v,0). For v=1: 0-1-0 is not a cycle (needs distinct vertices, len>=3).
    #
    # Let me use a cleaner approach. I'll define specific edges and verify cycles.
    #
    # Final construction (56 edges):
    edges = set()
    
    # Left path: 0-1-2-...-19
    for i in range(19):
        edges.add(frozenset({i, i+1}))
    
    # Right path: 20-21-22-...-39
    for i in range(19):
        edges.add(frozenset({20+i, 21+i}))
    
    # Hub edge
    edges.add(frozenset({0, 20}))
    
    # Chords: connect left vertices to right vertices
    # Chord pattern: (i, 20+i) for i in 1..17
    # This gives: cycle 0-1-...-i-(i,20+i)-(20+i)-20-(0,20)-0
    # Wait, that's not right. Let me trace:
    # 0 -> 1 -> 2 -> ... -> i -> (chord to 20+i) -> 20+i -> 20+i+1 -> ... -> 39? No.
    #
    # Let me trace more carefully:
    # 0 -> 1 -> ... -> i (dist i from 0 on left)
    # chord (i, 20+i)
    # 20+i -> 20+i-1 -> ... -> 20 (dist i from 20 on right, going backward)
    # 20 -> 0 (hub edge)
    # Total: i + 1 + i + 1 = 2i + 2
    # For i=1: len 4. For i=17: len 36.
    #
    # That gives even lengths 4, 6, 8, ..., 36. Missing odd lengths and 3, 37, 38, 39, 40.
    #
    # Let me add different chords for odd lengths.
    #
    # Chord (i, 20+i-1) for i in 2..18:
    # 0 -> 1 -> ... -> i (dist i)
    # chord (i, 20+i-1)
    # 20+i-1 -> 20+i-2 -> ... -> 20 (dist i-1)
    # 20 -> 0
    # Total: i + 1 + (i-1) + 1 = 2i + 1
    # For i=2: len 5. For i=18: len 37.
    #
    # So chords (i, 20+i) for i=1..17 give even lengths 4..36 (17 chords)
    # Chords (i, 20+i-1) for i=2..18 give odd lengths 5..37 (17 chords)
    # That's 34 chords, way too many.
    #
    # I need to be smarter. Let me use a mix.
    #
    # Better approach: Use a single set of chords that covers all lengths.
    #
    # Let me use chords (i, 20 + f(i)) for a carefully chosen function f.
    #
    # Cycle length via chord (i, 20+j): i + 1 + j + 1 = i + j + 2
    # I need i+j+2 to cover 3..40, i.e., i+j to cover 1..38.
    # With i in 1..19, j in 0..19.
    #
    # If I use chords (i, 20+i) for i=1..19: i+j = 2i, covers 2,4,...,38 -> lengths 4,6,...,40
    # If I use chords (i, 20+i-1) for i=2..19: i+j = 2i-1, covers 3,5,...,37 -> lengths 5,7,...,39
    #
    # Together these cover all lengths 4..40 except 3.
    # For length 3: need triangle. 0-1-20-0 with chord (1,20). That's chord (1, 20+0) = (1,20).
    #
    # So I need:
    # - Chord (1,20) for triangle (len 3)
    # - Chords (i, 20+i) for i=2..19 for even lengths 6..40 (18 chords)
    # - Chords (i, 20+i-1) for i=2..19 for odd lengths 5..39 (18 chords)
    #
    # Total chords: 1 + 18 + 18 = 37. Too many!
    #
    # I need to overlap. Let me find a smaller set.
    #
    # Key: I don't need every chord to be unique in its (i,j) pair. 
    # A chord (i, 20+j) gives cycle length i+j+2.
    # I need the set {i+j+2 : (i,20+j) in chords} to include 3..40.
    # i.e., {i+j : (i,20+j) in chords} to include 1..38.
    #
    # Minimum chords to cover sums 1..38 with i in 1..19, j in 0..19:
    # This is a covering problem. 
    #
    # Let me try: chords (i, 20+i) for i=1..19 give sums 2,4,...,38 (19 chords, even sums)
    # chords (i, 20+i-1) for i=2..19 give sums 3,5,...,37 (18 chords, odd sums >=3)
    # Sum 1: need i+j=1, so (1,0) = chord (1,20). Already have it in first set!
    #
    # So: chords (i, 20+i) for i=1..19 cover sums 2,4,...,38 AND sum 2.
    # Wait, i=1: sum=2. i=2: sum=4. ... i=19: sum=38.
    # Chords (i, 20+i-1) for i=2..19: i=2:sum=3, i=3:sum=5,...,i=19:sum=37.
    #
    # Combined: sums covered = {2,4,...,38} ∪ {3,5,...,37} = {2,3,4,...,38}
    # Missing sum 1! Need i+j=1: only (1,0). Chord (1,20).
    # Is (1,20) in my set? First set has (1,20+1)=(1,21). No.
    # Second set starts at i=2. No.
    #
    # So I need chord (1,20) additionally. Total: 19+18+1 = 38 chords. Still too many.
    #
    # Let me reduce. I don't need all 19 in the first set.
    #
    # Optimal covering: I need sums 1..38.
    # Use chords to hit as many sums as possible.
    #
    # Chord (i, 20+j) hits sum i+j.
    # 
    # Let me use a greedy/diagonal approach:
    # Chord (1,20): sum 1
    # Chord (1,21): sum 2
    # Chord (2,21): sum 3
    # Chord (2,22): sum 4
    # Chord (3,22): sum 5
    # ...
    # Chord (i, 20+i): sum 2i
    # Chord (i+1, 20+i): sum 2i+1
    #
    # Pattern: for k=1..19: chords (k, 20+k) [sum 2k] and (k+1, 20+k) [sum 2k+1]
    # But (k+1, 20+k) for k=19 would be (20, 39) which is already a path edge!
    #
    # Let me just use 17 chords cleverly.
    #
    # Actually, the problem says 16 chords suffice. Let me trust that and find them.
    #
    # Total edges target: 56. Base 39. Chords: 17.
    #
    # Let me just hardcode a verified construction.
    #
    # I'll use this chord set (17 chords):
    # For cycle length L, I need a chord enabling it.
    #
    # Chords: (1,20), (1,21), (2,21), (2,22), (3,22), (3,23), (4,23), (4,24),
    #         (5,24), (5,25), (6,25), (6,26), (7,26), (7,27), (8,27), (8,28), (9,28)
    #
    # Sums: 1,2, 3,4, 5,6, 7,8, 9,10, 11,12, 13,14, 15,16, 17
    # That covers sums 1..17, giving cycle lengths 3..19.
    #
    # For longer cycles, I need more. This only gives 17 chords for lengths 3-19.
    #
    # I think I'm overcomplicating this. Let me just use a known-good construction.
    #
    # Alternative: use a "double fan" from both hubs.
    #
    # Edges:
    # - 0 connected to 1,2,...,19 (left fan, 19 edges)
    # - 20 connected to 21,22,...,39 (right fan, 19 edges)
    # - (0,20) hub edge (1 edge)
    # - Cross chords to enable cycles
    #
    # Total base: 39 edges.
    #
    # With fan structure, cycle 0-1-2-...-k-0 has length k+1 if (k,0) exists.
    # (k,0) exists for all k in 1..19. So cycles of length 2..20 on left side!
    # Wait, cycle 0-1-0 is length 2, invalid. Cycle 0-1-2-0: needs (1,2) and (2,0).
    # (2,0) exists (fan). (1,2) does NOT exist in fan-only.
    #
    # In a pure fan, the only cycles are triangles 0-i-j-0 if (i,j) exists.
    #
    # So I need path edges too. Let me combine:
    #
    # Left: path 0-1-2-...-19 (19 edges) + fan chords from 0 to 3,4,...,19 (17 chords)
    # Right: path 20-21-...-39 (19 edges) + fan chords from 20 to 23,...,39 (17 chords)
    # Hub: (0,20) (1 edge)
    #
    # Total: 19+17+19+17+1 = 73. Too many.
    #
    # I'll just go with a straightforward construction and verify it works.
    # Let me use 56 edges with a specific pattern.
    
    # Reset
    edges = set()
    
    # Left path: 0-1-...-19
    for i in range(19):
        edges.add(frozenset({i, i+1}))
    
    # Right path: 20-21-...-39
    for i in range(19):
        edges.add(frozenset({20+i, 21+i}))
    
    # Hub
    edges.add(frozenset({0, 20}))
    
    # 17 cross chords
    # Chord (i, 20+i) for i in 1..17: enables cycle of length 2i+2
    # 0-1-...-i-(i,20+i)-(20+i)-...-20-0: length i + 1 + i + 1 = 2i+2
    # i=1:4, i=2:6, ..., i=17:36
    for i in range(1, 18):
        edges.add(frozenset({i, 20+i}))
    
    # That's 39 + 17 = 56 edges.
    # Cycles enabled: lengths 4,6,8,...,36 (even)
    # Missing: 3,5,7,...,35,37,38,39,40
    
    # I need more chords for odd lengths and 37-40.
    # But I'm already at 56. I need to replace some chords.
    
    # Let me redesign with 56 edges total but better chord placement.
    
    edges = set()
    
    # Left path: 0-1-...-19 (19 edges)
    for i in range(19):
        edges.add(frozenset({i, i+1}))
    
    # Right path: 20-21-...-39 (19 edges)
    for i in range(19):
        edges.add(frozenset({20+i, 21+i}))
    
    # Hub: (0,20) (1 edge)
    edges.add(frozenset({0, 20}))
    
    # Now 39 edges. Need 17 more.
    #
    # Chord strategy:
    # (1,20): triangle 0-1-20-0, len 3
    # (1,21): 0-1-21-20-0, len 4
    # (2,21): 0-1-2-21-20-0, len 5
    # (2,22): 0-1-2-22-21-20-0, len 6
    # (3,22): 0-1-2-3-22-21-20-0, len 7
    # (3,23): len 8
    # (4,23): len 9
    # (4,24): len 10
    # (5,24): len 11
    # (5,25): len 12
    # (6,25): len 13
    # (6,26): len 14
    # (7,26): len 15
    # (7,27): len 16
    # (8,27): len 17
    # (8,28): len 18
    # (9,28): len 19
    #
    # That's 17 chords covering lengths 3-19.
    # Still missing 20-40.
    
    # I need a different approach for longer cycles.
    #
    # For longer cycles, use the full right path:
    # 0-1-...-i-(chord to 20+j)-(20+j)-...-39-... back? No, need to return to 0.
    #
    # Use chord (i, 39) for various i:
    # 0-1-...-i-(i,39)-39-38-...-20-0: length i + 1 + 20 + 1 = i + 22
    # i=1:23, i=2:24, ..., i=18:40
    #
    # And chord (i, 38):
    # 0-1-...-i-(i,38)-38-37-...-20-0: length i + 1 + 19 + 1 = i + 21
    # i=1:22, i=2:23, ..., i=17:38
    #
    # Let me combine:
    # Chords for lengths 3-19: (1,20),(1,21),(2,21),(2,22),...,(9,28) [17 chords]
    # Chords for lengths 20-40: need 21 more lengths.
    #
    # This is getting complex. Let me just use a larger edge count that I can verify.
    #
    # Actually, let me just go with a simple verified construction, even if not optimal.
    # The seed has 77 edges (score 37). If I can get to 60 edges (score 20), that's an improvement.
    #
    # Simple construction:
    # - Cycle 0-1-2-...-39-0 (40 edges)
    # - Chords from 0 to all vertices: (0,2),(0,3),...,(0,38) (37 chords, (0,1) and (0,39) already in cycle)
    # Total: 40 + 37 = 77. Same as seed!
    #
    # Better:
    # - Path 0-1-...-39 (39 edges)
    # - Chords (0,k) for k=2..39 (38 chords, (0,1) is in path)
    # Total: 39 + 38 = 77.
    #
    # Let me try:
    # - Two paths meeting at hub: 0-1-...-19-20-21-...-39 (39 edges, it's one path)
    # - Plus chords to make it a cycle and add shortcuts
    #
    # Path 0-1-...-39: 39 edges. Add (0,39): 40 edges (now a cycle).
    # Add chords (0,k) for k=2..38: 37 chords. Total: 77.
    #
    # I'll just go with the two-hub + 17 chords = 56 edge construction and verify cycles.
    # If it doesn't cover all lengths, I'll add edges.
    
    # Let me just do a clean construction:
    edges = set()
    
    # Cycle: 0-1-2-...-39-0
    for i in range(40):
        edges.add(frozenset({i, (i+1) % 40}))
    
    # Chords from 0 to enable all cycle lengths
    # Cycle of length k: 0-1-2-...-(k-1)-0, needs chord (0,k-1)
    # For k=3: chord (0,2)
    # For k=4: chord (0,3)
    # ...
    # For k=39: chord (0,38)
    # For k=40: already the cycle
    for k in range(3, 40):
        chord = frozenset({0, k-1})
        if chord not in edges:
            edges.add(chord)
    
    # Count: 40 cycle edges + chords (0,2)..(0,38) = 37 chords, but (0,1) and (0,39) are in cycle
    # Chords added: (0,2),(0,3),...,(0,38) = 37 chords
    # Total: 40 + 37 = 77. Same as before.
    
    # This is not improving. Let me think differently.
    
    # The key is to share edges between multiple cycles.
    #
    # Best approach: use the two-hub structure with minimal chords.
    #
    # I'll go with a construction I can verify has 56 edges.
    
    edges = set()
    
    # Left path: 0-1-...-19
    for i in range(19):
        edges.add(frozenset({i, i+1}))
    
    # Right path: 20-21-...-39
    for i in range(19):
        edges.add(frozenset({20+i, 21+i}))
    
    # Hub
    edges.add(frozenset({0, 20}))
    
    # 17 chords: (i, 20+i) for i=1..17
    for i in range(1, 18):
        edges.add(frozenset({i, 20+i}))
    
    # Total: 39 + 17 = 56 edges
    
    # Now construct cycles:
    cycles = []
    
    # For length L, find a cycle.
    # Even lengths 4-36: use chord (i, 20+i) where 2i+2 = L, i = (L-2)/2
    # Odd lengths and others: need alternative.
    
    # Let me check what cycles I can form:
    # With chord (i, 20+i): cycle 0-1-...-i-(20+i)-...-20-0
    # Path from 20+i back to 20: (20+i)-(20+i-1)-...-20, length i
    # Total: i (left) + 1 (chord) + i (right back) + 1 (hub) = 2i+2
    
    # So I have cycles of length 4,6,8,...,36.
    # For odd lengths, I can use: 0-1-...-i-(20+i)-...-20-1-0? No, (1,0) is in path.
    # Actually 0-1-...-i-(20+i)-20-(0,20)-0 is the even one.
    # For odd: 0-1-...-i-(20+i)-...-(20+i-1)-...-20-0? Same thing.
    
    # I need chords for odd lengths. Let me add a few more.
    # But I'm at 56. Let me replace some even chords with odd ones.
    
    # Actually, let me just go with 57 edges to be safe, or find the right 56.
    
    # Let me use this construction:
    edges = set()
    for i in range(19):
        edges.add(frozenset({i, i+1}))
    for i in range(19):
        edges.add(frozenset({20+i, 21+i}))
    edges.add(frozenset({0, 20}))
    
    # Chords: (1,20) for len 3
    edges.add(frozenset({1, 20}))
    # (i, 20+i) for i=2..17 for even lengths 6..36
    for i in range(2, 18):
        edges.add(frozenset({i, 20+i}))
    # (i, 20+i-1) for i=3..18 for odd lengths 7..37
    for i in range(3, 19):
        edges.add(frozenset({i, 20+i-1}))
    
    # Count: 39 + 1 + 16 + 16 = 72. Too many.
    
    # I'll just go with a working construction, even if not minimal.
    # Let me use 60 edges.
    
    edges = set()
    for i in range(19):
        edges.add(frozenset({i, i+1}))
    for i in range(19):
        edges.add(frozenset({20+i, 21+i}))
    edges.add(frozenset({0, 20}))
    
    # 21 chords to cover all lengths
    # (i, 20+i) for i=1..21? But i max is 19.
    # (i, 20+i) for i=1..19: 19 chords, even lengths 4..40
    for i in range(1, 20):
        edges.add(frozenset({i, 20+i}))
    # (i, 20+i-1) for i=2..19: 18 chords, odd lengths 5..39
    # But I only need odd lengths not covered. Even covers 4,6,...,40.
    # Need: 3,5,7,...,39
    # (1,20): len 3
    edges.add(frozenset({1, 20}))
    # (i, 20+i-1) for i=2..19: len 5,7,...,39
    for i in range(2, 20):
        edges.add(frozenset({i, 20+i-1}))
    
    # Total: 39 + 19 + 1 + 18 = 77. Back to seed.
    
    # I give up on optimization and just return a valid construction.
    # Let me use the cycle + chords approach with 77 edges, which is valid.
    
    edges = set()
    for i in range(40):
        edges.add(frozenset({i, (i+1) % 40}))
    for k in range(2, 39):
        edges.add(frozenset({0, k}))
    
    edge_list = [tuple(sorted(e)) for e in edges]
    
    cycles = []
    for L in range(3, 41):
        if L == 40:
            cycles.append(list(range(40)))
        else:
            cycles.append(list(range(L)))
    
    return {'edges': edge_list, 'cycles': cycles}
