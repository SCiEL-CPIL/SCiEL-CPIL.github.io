import time

def construct():
    """
    Constructs a sparse pancyclic graph on 40 vertices.
    Returns a dictionary with keys 'edges' and 'cycles'.
    'edges' is a list of undirected edge pairs (u, v) with 0 <= u < v < 40.
    'cycles' is a list of 38 cycles (lengths 3 to 40), each represented as a list of vertices.
    """
    N = 40
    start_time = time.time()

    # Safety margin: return early if we exceed 85 seconds
    # The construction is O(N), so this is unlikely to be triggered.

    edges = set()

    # Hamiltonian cycle on 0..39: edges (i, i+1) mod 40
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))

    # We need cycles of every length 3..40.
    # Hamiltonian cycle gives length 40.
    # Remaining lengths: 3..39 (37 lengths).
    #
    # Strategy: use chords from vertex 0 to cover these lengths.
    # A chord (0, k) can give a cycle of length k+1 (path 0-1-2-...-k-0) 
    # and potentially 41-k (path 0-39-38-...-k-0).
    #
    # To minimize edges, we want each chord to cover as many lengths as possible.
    # The optimal approach is to use chords that cover 2 lengths each if possible,
    # and 1 length for the odd one out.
    #
    # Lengths to cover: 3, 4, 5, ..., 39
    #
    # If we use chord (0, k), we get length k+1.
    # If k is even, we might also get length 41-k from the "other side".
    # But actually, the path 0-39-...-k has length 41-k only if k >= 2.
    # For the cycle to be simple, we need the vertices on the path to be distinct.
    # The path 0-39-38-...-k uses vertices {0, k, k+1, ..., 39}.
    # The path 0-1-...-k uses vertices {0, 1, ..., k}.
    #
    # Let's assign chords to cover specific lengths.
    # We need 37 lengths.
    # If we use 19 chords, and each covers 2 lengths, we cover 38. That's more than 37.
    # Wait, 19 * 2 = 38. So 18 chords covering 2 lengths and 1 chord covering 1 length?
    # 18*2 + 1 = 37. Yes.
    #
    # Which chords cover which lengths?
    # Length L. 
    # Case 1: Cycle is 0-1-...-L-1-0. Requires chord (0, L-1). Length of chord is L-1.
    #          Vertices: 0, 1, ..., L-1. Distinct if L-1 < 40.
    # Case 2: Cycle is 0-39-...-L-1-0? No, that's not right.
    #          The path from 0 to L-1 via the "long" way around the Hamiltonian cycle
    #          is 0-39-38-...-(L-1). The number of edges is 40 - (L-1) = 41-L.
    #          The number of vertices is 41-L + 1 = 42-L?
    #          Let's count: 0, 39, ..., L-1.
    #          Number of vertices = 39 - (L-1) + 1 = 41 - L + 1 = 42 - L?
    #          Let's check L=39. Path 0-39. Vertices 0, 39. Length 2? No, cycle length is number of vertices.
    #          Cycle: 0-39-0. Length 2? No, edges are (0,39) and (39,0)? No, undirected.
    #          Cycle: 0-39-...?
    #          For L=39, we need a cycle of length 39.
    #          Path 0-39-...? The Hamiltonian cycle minus one vertex.
    #          Remove vertex v. The remaining path is 0-1-...-v-1 and v+1-...-39-0.
    #          To make a cycle, we need to connect the ends.
    #          Actually, simpler:
    #          The cycle formed by chord (0, k) and the path 0-1-...-k has length k+1.
    #          The cycle formed by chord (0, k) and the path 0-39-...-k has length (40-k) + 1?
    #          Path 0-39-38-...-k has 40-k edges?
    #          0 to 39 is 1 edge. 39 to 38 is 1 edge. ... k+1 to k is 1 edge.
    #          Number of edges = 1 + (39 - k) = 40 - k.
    #          Number of vertices = 40 - k + 1 = 41 - k.
    #          So chord (0, k) covers length k+1 and 41-k.
    #
    # We need to cover {3, ..., 39}.
    # Let's pick k such that k+1 and 41-k are in this set.
    # k+1 >= 3 => k >= 2.
    # 41-k >= 3 => k <= 38.
    # Also k < 40.
    #
    # We want to choose a set of k's such that the union of {k+1, 41-k} is {3, ..., 39}.
    # Note that k+1 = 41-k => 2k = 40 => k = 20.
    # So k=20 gives length 21 (twice? no, just one cycle of length 21).
    # Wait, the cycle vertices for k+1 and 41-k are disjoint?
    # Cycle 1: 0, 1, ..., k. Vertices {0..k}.
    # Cycle 2: 0, 39, ..., k. Vertices {k..39, 0}.
    # They share 0 and k.
    # If k=20, Cycle 1 is 0..20 (len 21). Cycle 2 is 0..20 (len 21). Same cycle!
    # So k=20 covers only length 21.
    #
    # For k != 20, k+1 != 41-k.
    # So each k != 20 covers two distinct lengths.
    #
    # We need to cover 37 lengths.
    # One length must be 21 (covered by k=20 alone).
    # The other 36 lengths must be covered by 18 chords (each covering 2).
    # Total chords = 1 (for k=20) + 18 = 19.
    #
    # Which 18 chords?
    # The lengths are 3..39 excluding 21.
    # Set of lengths: {3, ..., 20, 22, ..., 39}.
    # Each chord k (where k != 20) covers {k+1, 41-k}.
    # Let's list the pairs:
    # k=2 -> {3, 39}
    # k=3 -> {4, 38}
    # ...
    # k=19 -> {20, 22}
    # k=21 -> {22, 20} (Same pair as k=19, but different cycle?)
    # Wait, if we use k=19, we get cycles of length 20 and 22.
    # If we use k=21, we get cycles of length 22 and 20.
    # We only need ONE cycle of length 20 and ONE cycle of length 22.
    # So using either k=19 or k=21 covers both lengths 20 and 22.
    # We should pick one of them to avoid redundancy?
    # No, we just need the union to be the target set.
    # If we pick k=19, we cover {20, 22}.
    # If we pick k=21, we cover {22, 20}.
    # It doesn't matter which we pick, but we only need to pick ONE chord to cover these two lengths.
    #
    # Let's check the pairs for k in {2, ..., 19} U {21, ..., 38}?
    # Actually, k and 40-k are symmetric?
    # Chord (0, k) is the same as chord (0, 40-k)? No, vertex 40-k is different from k.
    # But chord (0, k) covers {k+1, 41-k}.
    # Chord (0, 40-k) covers {41-k, k+1}.
    # So chord (0, k) and chord (0, 40-k) cover the EXACT same set of lengths!
    # This is a crucial insight.
    #
    # So, we don't need both k and 40-k. We just need to pick one representative from each pair {k, 40-k}.
    # And we need to cover all lengths.
    #
    # The pairs of k (where k covers lengths L1, L2) are determined by the set of lengths.
    # Lengths to cover: {3, ..., 39} \ {21} = 36 lengths.
    # These form 18 pairs of lengths:
    # {3, 39}, {4, 38}, ..., {20, 22}.
    #
    # For the pair {3, 39}, we need a chord k such that {k+1, 41-k} = {3, 39}.
    # k+1=3 => k=2. 41-2=39. OK.
    # Or k+1=39 => k=38. 41-38=3. OK.
    # So we can pick k=2 or k=38. Let's pick k=2.
    #
    # For the pair {4, 38}, we need k such that {k+1, 41-k} = {4, 38}.
    # k=3 or k=37. Pick k=3.
    #
    # ...
    # For the pair {20, 22}, we need k such that {k+1, 41-k} = {20, 22}.
    # k=19 (20, 22) or k=21 (22, 20). Pick k=19.
    #
    # So the chords are:
    # k = 2, 3, ..., 19. (18 chords)
    # Plus k = 20 (for length 21).
    #
    # Total chords: 19.
    # Edges: 40 (Hamiltonian) + 19 = 59.
    #
    # This confirms the current solution is optimal.
    # We just need to implement it correctly and efficiently.

    for k in range(2, 20):  # 2 to 19
        edges.add((0, k))

    edges.add((0, 20))

    edge_list = list(edges)
    cycles = []

    # Length 3
    cycles.append([0, 1, 2])
    # Length 4
    cycles.append([0, 1, 2, 3])
    # ...
    # We can generate them systematically.
    
    for L in range(3, N + 1):
        if time.time() - start_time > 85.0:
            break

        if L == N:
            cyc = list(range(N))
        elif L == 21:
            # Chord (0, 20). Cycle 0-1-...-20-0.
            cyc = list(range(21))
        else:
            # L in 3..20 or 22..39
            # If L <= 20: Chord k = L-1. Cycle 0-1-...-L-1-0.
            # If L > 20: Let k = L-1? No.
            # If L > 20, L is in {22..39}.
            # We used chord k for L if k+1 = L or 41-k = L.
            # Our chords are k=2..19.
            # For k in 2..19, the lengths are k+1 (3..20) and 41-k (39..22).
            # So if L <= 20, the chord is k = L-1.
            # The cycle is 0-1-...-L-1-0.
            # If L > 20, the chord is k such that 41-k = L => k = 41-L.
            # The cycle is 0-39-...-k-0.
            # Let's verify:
            # If L=39, k=2. Cycle 0-39-...-2-0.
            # Vertices: 0, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2.
            # Wait, the path is 0-39-38-...-2.
            # Vertices: 0, 39, 38, ..., 2.
            # Count: 1 (for 0) + (39 - 2 + 1) = 1 + 38 = 39. Correct.
            # Edges: (0,39), (39,38), ..., (3,2), (2,0).
            # (0,39) is in Hamiltonian.
            # (39,38)...(3,2) are in Hamiltonian.
            # (2,0) is the chord.
            # So the cycle is valid.

            if L <= 20:
                cyc = list(range(L))
            else:
                k = 41 - L
                # Cycle: 0, 39, 38, ..., k
                # Note: range(39, k-1, -1) gives 39, ..., k.
                cyc = [0] + list(range(39, k - 1, -1))
        
        cycles.append(cyc)

    while len(cycles) < N - 2:
        L = len(cycles) + 3
        if L == N:
            cyc = list(range(N))
        elif L == 21:
            cyc = list(range(21))
        elif L <= 20:
            cyc = list(range(L))
        else:
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        cycles.append(cyc)

    return {'edges': edge_list, 'cycles': cycles}
