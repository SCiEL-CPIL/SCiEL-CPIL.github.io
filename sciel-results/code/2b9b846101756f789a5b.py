N = 40

def construct():
    # Build a sparse pancyclic graph on 40 vertices using a "double-fan" design.
    # Vertices are arranged in a ring 0..39.
    # We add chords from two hub vertices (0 and 20) to create many cycle lengths.
    #
    # Edge set:
    #   Ring edges: (i, i+1 mod 40) for all i -> 40 edges
    #   Chords from 0: (0, k) for k in {2, 3, ..., 19} -> 18 edges
    #   Chords from 20: (20, k) for k in {22, 23, ..., 39, 0} -> 18 edges
    #     Note: (20, 0) is the diameter chord, already useful.
    #   Total: 40 + 18 + 18 = 76 edges
    #   Score = 76 - 40 = 36
    #
    # Cycle certificates for lengths 3..40:
    #   Length L uses vertices from the ring and possibly hub chords.
    #   For L in 3..20: cycle 0-1-2-...-(L-1)-0 uses chord (0, L-1) for L-1 >= 2,
    #     and ring edges otherwise. For L=3: 0-1-2-0 needs edge (2,0) which is chord (0,2). Good.
    #     For L=20: 0-1-...-19-0 needs chord (0,19). Good.
    #   For L in 21..39: use the "far" side. Cycle: 20-21-22-...-(20+L-1 mod 40)-20
    #     But we need chord (20, last). Let's design more carefully.
    #
    # Better cycle construction:
    #   L = 3..20: cycle = [0, 1, 2, ..., L-1]. Edges: ring edges (i,i+1) for i=0..L-2, and chord (0, L-1).
    #     Valid for L>=3 since chord (0, L-1) exists for L-1 in 2..19.
    #   L = 21: cycle = [0, 1, 2, ..., 20]. Edges: ring edges 0-1-...-20, and chord (0,20)? 
    #     Wait, (0,20) is a chord from 20's set. Yes, (20,0) is in the chord set. Good.
    #   L = 22..40: cycle = [20, 21, 22, ..., (20+L-1) mod 40]. 
    #     For L=22: [20,21,...,41 mod 40=1]. That's 20,21,...,39,0,1. Length 22.
    #     Edges: ring edges along the path, and chord (20, 1). But (20,1) is not in our chord set!
    #
    # Let me reconsider. Use a different strategy for the upper range.
    #   For L in 21..39: cycle = [0, 39, 38, ..., (41-L), 1, 2, ..., 0]? No, too complex.
    #
    # Simpler: For L in 22..40, use cycle starting at 0 going the "long way" around.
    #   Cycle for L: [0, 39, 38, ..., (40-L), 1, 2, ..., 0]? 
    #   Length = (39 - (40-L) + 1) + (1 - 0 + 1) = L + 1? No.
    #
    # Let's just use: for L in 21..39, cycle = [0, 1, 2, ..., L-1] but that only works up to L=20 with chord (0,L-1).
    #
    # Alternative: Add chords (0,k) for k=2..39 (38 chords) -> 40+38=78 edges, score 38.
    #   Then all cycles 3..40 use [0,1,...,L-1] with chord (0,L-1).
    #   That's simpler and valid. Score = 38.
    #
    # Can we do better? The original had score 19 (59 edges - 40 = 19). Wait, original:
    #   40 ring + 19 chords = 59 edges. Score = 59-40 = 19.
    #   The current program scores 1.9474, which means value = 19, seed = ~9.76? No, score = seed/value.
    #
    # I'll go with the simpler design: ring + chords (0,k) for k=2..39.
    # 40 + 38 = 78 edges. Score = 38.
    #
    # Actually, let's try to be smarter. Use chords (0,k) for k=2..20 and (20,k) for k=22..39.
    # 40 + 19 + 18 = 77 edges. Score = 37.
    # Cycles:
    #   L=3..21: [0,1,...,L-1], chord (0,L-1) for L-1<=20. For L=21, chord (0,20). Good.
    #   L=22..40: [20,21,...,20+L-1 mod 40], chord (20, last). 
    #     L=22: last = 21+21=42 mod 40=2. Chord (20,2)? Not in set.
    #
    # I'll stick with the straightforward approach: ring + all chords from 0.
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    for k in range(2, N):
        edges.add((0, k))
    
    edge_list = sorted(tuple(sorted(e)) for e in edges)
    
    cycles = []
    for L in range(3, N + 1):
        cyc = list(range(L))
        cycles.append(cyc)
    
    return {'edges': edge_list, 'cycles': cycles}
