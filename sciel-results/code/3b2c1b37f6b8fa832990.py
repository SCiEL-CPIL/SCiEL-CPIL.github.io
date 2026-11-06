"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Two cliques K_20 on vertices 0..19 and 20..39, connected by perfect matching
    # This gives us:
    # - All edges within clique A: C(20,2) = 190 edges
    # - All edges within clique B: C(20,2) = 190 edges
    # - Perfect matching between corresponding vertices: 20 edges
    # Total: 400 edges
    # Score = 400 - 40 = 360 (worse than seed's 79)
    # 
    # Wait, let me reconsider. The seed has 79 edges. We need to do better.
    # 
    # Let me think about this differently. The task says "Minimize edge count minus 40."
    # So we want as few edges as possible while still having cycles of every length 3..40.
    #
    # A pancyclic graph on n vertices needs at least n edges (a Hamiltonian cycle gives 
    # cycles of all lengths? No, a Hamiltonian cycle only gives one cycle of length n and 
    # doesn't guarantee cycles of all other lengths).
    #
    # Actually, a cycle of length n is pancyclic if and only if... no, a single n-cycle 
    # only contains that one cycle. We need additional edges.
    #
    # The seed construction: star from vertex 0 to 1..39 (39 edges) plus path 1-2-...-39 (38 edges)
    # Total: 77 edges. Wait, let me recount.
    # [(0,i) for i in range(1,N)] = 39 edges
    # [(i,i+1) for i in range(1,N-1)] = 38 edges
    # Total = 77 edges. Score = 77 - 40 = 37.
    #
    # But the cycles in the seed are [list(range(k)) for k in range(3,N+1)]
    # For k=3: [0,1,2] - needs edges (0,1), (1,2), (2,0). (0,1) exists, (1,2) exists, but (2,0)? Yes, (0,2) exists.
    # For k=4: [0,1,2,3] - needs (0,1),(1,2),(2,3),(3,0). All exist.
    # For k=40: [0,1,...,39] - needs (0,1),(1,2),...,(38,39),(39,0). (39,0) exists since (0,39) is in the star.
    # 
    # So the seed works with 77 edges.
    #
    # Can we do better? A tree on 40 vertices has 39 edges. Can a tree be pancyclic? No, trees have no cycles.
    # We need at least some cycles. 
    #
    # Let me think about what's the minimum number of edges for a pancyclic graph on 40 vertices.
    # 
    # Actually, let me try a different approach. Use a Hamiltonian cycle (40 edges) plus some chords to create shorter cycles.
    # 
    # With a Hamiltonian cycle 0-1-2-...-39-0, we have a cycle of length 40.
    # To get a cycle of length k, we can use a chord. For example, if we add edge (0, k-1), we get a cycle of length k: 0-1-2-...-(k-1)-0.
    # 
    # So if we add chords (0,2), (0,3), ..., (0,38), we get cycles of lengths 3,4,...,39.
    # Plus the Hamiltonian cycle gives length 40.
    # 
    # Edges: 40 (Hamiltonian cycle) + 37 (chords from 0 to 2,3,...,38) = 77 edges.
    # Same as the seed!
    #
    # Can we do better? What if we share chords more efficiently?
    #
    # Actually, the seed is already pretty optimal. Let me verify: 77 edges, score 37.
    #
    # Is there a way to get fewer than 77 edges? 
    # 
    # Consider: we need cycles of lengths 3,4,5,...,40. That's 38 cycles.
    # Each cycle of length k uses k edges. But edges can be shared.
    # 
    # The minimum number of edges in a pancyclic graph... I believe it's known that n + floor(n/2) - 2 or something similar might be achievable, but let me think more carefully.
    #
    # Actually, for the seed approach: star + path gives 77. 
    # Alternative: use a "ladder" or other structure.
    #
    # Let me try: take a cycle 0-1-2-...-39-0 (40 edges). Then add chords to create shorter cycles.
    # To create a cycle of length k, we need a chord that shortcuts the path. 
    # If we add chord (i, j) where j > i, we create a cycle of length (j - i) + 1 (going forward from i to j along the cycle) and a cycle of length (40 - (j - i)) + 1... no wait.
    # 
    # If the cycle is 0-1-2-...-39-0, and we add chord (a, b) with b > a, then:
    # - Cycle 1: a, a+1, ..., b, a (length b-a+1)
    # - Cycle 2: a, b, b+1, ..., 39, 0, 1, ..., a-1, a (length 40 - (b-a) + 1 = 41 - (b-a))
    # 
    # So each chord creates two cycles. We need cycles of lengths 3 through 40.
    # The Hamiltonian cycle gives length 40.
    # We need lengths 3,4,...,39.
    # 
    # If we add chord (0, k-1), we get cycle of length k (0,1,...,k-1,0) and cycle of length 40-k+1 = 41-k.
    # So chord (0, k-1) gives cycles of lengths k and 41-k.
    # 
    # For k=3: chord (0,2) gives cycles of length 3 and 38.
    # For k=4: chord (0,3) gives cycles of length 4 and 37.
    # ...
    # For k=20: chord (0,19) gives cycles of length 20 and 21.
    # 
    # So chords (0,2), (0,3), ..., (0,19) give cycles of lengths {3,4,...,20} ∪ {38,37,...,21} = {3,4,...,38}.
    # Plus the Hamiltonian cycle gives 39? No, the Hamiltonian cycle is length 40.
    # We're missing cycle of length 39.
    # 
    # Chord (0,19) gives 20 and 21. Chord (0,18) gives 19 and 22.
    # So we have 3..20 and 21..38. Missing 39.
    # 
    # To get 39, we can add chord (0,38): gives cycle of length 39 (0,1,...,38,0) and cycle of length 2 (which is not a valid cycle, so just 39).
    # Wait, cycle of length 2 is not valid. So chord (0,38) gives only the cycle of length 39.
    # 
    # So total chords: (0,2), (0,3), ..., (0,19) = 18 chords, plus (0,38) = 19 chords.
    # Total edges: 40 + 19 = 59. Score = 59 - 40 = 19.
    # 
    # That's much better than 77!
    #
    # Let me verify: 
    # - Cycle of length 3: [0,1,2] with edges (0,1),(1,2),(2,0). (2,0) is the chord. ✓
    # - Cycle of length 4: [0,1,2,3] with edges (0,1),(1,2),(2,3),(3,0). (3,0) is the chord. ✓
    # - ...
    # - Cycle of length 20: [0,1,...,19] with chord (0,19). ✓
    # - Cycle of length 21: [0,19,20,...,39,1,...,18,0]? No, that's not right.
    #   The other cycle from chord (0,19) is: 0,19,20,21,...,39,0. That's length 21. ✓
    # - Cycle of length 22: from chord (0,18): 0,18,19,20,...,39,0. Length 22. ✓
    # - ...
    # - Cycle of length 38: from chord (0,3): 0,3,4,...,39,0. Length 37? No.
    #   Wait, let me recompute. Chord (0,k-1) gives:
    #   Cycle A: 0,1,...,k-1,0 (length k)
    #   Cycle B: 0,k-1,k,...,39,0 (length 40 - (k-1) + 1 = 42-k... no)
    #   
    #   Path from 0 to k-1 going forward: 0,1,...,k-1. Length k-1 edges, k vertices.
    #   Path from k-1 to 0 going forward: k-1,k,...,39,0. Length 40-(k-1) = 41-k edges, 42-k vertices.
    #   So cycle B has length 42-k.
    #   
    #   For k=3: cycle B length = 39. 
    #   For k=4: cycle B length = 38.
    #   ...
    #   For k=20: cycle B length = 22.
    #   For k=21: we don't have chord (0,20) in our set (we only go up to (0,19)).
    #   
    #   So from chords (0,2) through (0,19), we get cycles of lengths:
    #   {3,4,...,20} ∪ {39,38,...,22} = {3,4,...,20} ∪ {22,...,39}
    #   Missing: 21.
    #   
    #   To get 21, add chord (0,20): gives cycles of length 21 and 42-21=21. So just one cycle of length 21 (both sides are length 21).
    #   
    #   So we need chords (0,2), (0,3), ..., (0,20) = 19 chords.
    #   Total edges: 40 + 19 = 59.
    #   
    #   Cycles covered: 3..20 (from forward paths), 21 (from chord (0,20)), 22..39 (from backward paths of chords (0,19) down to (0,3)).
    #   Wait, let me list them:
    #   Chord (0,k-1) for k=3..20:
    #   - Forward cycle length: k
    #   - Backward cycle length: 42-k
    #   
    #   k=3: forward 3, backward 39
    #   k=4: forward 4, backward 38
    #   k=5: forward 5, backward 37
    #   ...
    #   k=20: forward 20, backward 22
    #   k=21: forward 21, backward 21 (same cycle)
    #   
    #   So we get: {3,4,...,20} ∪ {22,23,...,39} ∪ {21} = {3,4,...,39}. 
    #   Plus the Hamiltonian cycle of length 40.
    #   All lengths 3..40 covered! ✓
    #
    # Total edges: 59. Score: 19.
    #
    # Can we do even better? What if we use chords not all from vertex 0?
    # 
    # Each chord creates at most 2 cycles. We need 38 distinct cycle lengths (3..40, excluding 40 which is the Hamiltonian cycle). So we need at least ceil(38/2) = 19 chords.
    # So 59 edges seems to be the minimum with this approach.
    #
    # But wait, can we use the Hamiltonian cycle edges more cleverly? The Hamiltonian cycle itself is 40 edges. We need 38 more cycle lengths. Each chord adds at most 2 new cycle lengths. So minimum chords = 19, minimum total edges = 59.
    #
    # Let me implement this.

    edges = set()
    # Hamiltonian cycle: 0-1-2-...-39-0
    for i in range(N):
        edges.add((min(i, (i+1) % N), max(i, (i+1) % N)))
    
    # Chords from 0 to k-1 for k=3..21 (i.e., chords (0,2), (0,3), ..., (0,20))
    for k in range(3, 22):
        edges.add((0, k-1))
    
    edges_list = sorted(edges)
    
    # Now construct cycles
    cycles = []
    for k in range(3, N+1):
        if k == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        else:
            # For cycle of length k, use chord (0, k-1) if k <= 21, else use the backward cycle
            if k <= 21:
                # Forward cycle: 0, 1, 2, ..., k-1, 0
                cycles.append(list(range(k)))
            else:
                # Backward cycle from chord (0, 42-k-1) = (0, 41-k)
                # This gives cycle: 0, 41-k, 42-k, ..., 39, 0
                # Length = 40 - (41-k) + 1 = k
                # Vertices: 0, 41-k, 42-k, ..., 39
                start = 41 - k
                cyc = [0] + list(range(start, N))
                cycles.append(cyc)
    
    return {'edges': edges_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
