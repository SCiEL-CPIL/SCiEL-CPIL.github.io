N = 40

# EVOLVE-BLOCK-START
def construct():
    # We build a sparse pancyclic graph on 40 vertices.
    # Start with a Hamiltonian cycle (40 edges) and add chords to cover all cycle lengths 3..40.
    # The current design uses chords (0, j) for j in 2..20, giving 19 extra edges = 59 total.
    # Score = edges - 40 = 19.
    #
    # To improve, we try to reduce the number of chords while still covering all lengths 3..40.
    # Key insight: with chords (0, j) for j in S, we get cycles of length j (going 0-1-2-...-j-0)
    # and cycles of length 41-j (going 0-39-38-...-(41-j)-0). So each chord j covers lengths j and 41-j.
    # We need to cover all lengths in {3, 4, ..., 40}.
    # Length 40 is covered by the Hamiltonian cycle itself.
    # So we need to cover {3, 4, ..., 39}.
    # Chord j covers {j, 41-j}. For j in 2..20, this gives:
    #   j=2: {2, 39} -> covers 39 (2 is extra)
    #   j=3: {3, 38}
    #   j=4: {4, 37}
    #   ...
    #   j=19: {19, 22}
    #   j=20: {20, 21}
    # Union: {2,3,...,20} U {21,22,...,39} = {2,3,...,39}
    # We need {3,...,39}, so we can drop j=2 (which gives {2,39}) but then we lose 39.
    # Alternatively, keep j=2 and see if we can drop another chord.
    # If we drop j=3, we lose {3, 38}. Is 38 covered elsewhere? No.
    # So we need all of j=2..20 to cover {2..39}, and we need {3..39}.
    # Can we drop j=2? Then we lose 39. Can we cover 39 another way?
    # 39 = 41-2, so only j=2 gives 39. So we must keep j=2.
    # Can we drop any other? Each j in 3..20 gives a unique pair {j, 41-j} not covered by others.
    # So 19 chords are needed in this family.
    #
    # Alternative: use a different set of chords not all from vertex 0.
    # But the directive says to keep the design family. Let's try a small optimization:
    # Instead of chords (0, j) for j=2..20, can we use fewer by choosing chords more cleverly?
    # Actually, let's reconsider: do we really need chord (0,2)? It gives cycle length 2 (invalid, we need >=3) and 39.
    # We need length 39. The only way in this family is via chord (0,2).
    # What if we use a different approach for length 39? A cycle of length 39 uses 39 vertices, omitting 1.
    # In the HC 0-1-2-...-39-0, removing vertex v and connecting its neighbors gives a cycle of length 39.
    # E.g., remove vertex 1: cycle 0-39-38-...-2-0 requires edge (0,2). That's chord (0,2).
    # Remove vertex 0: cycle 1-2-...-39-1 requires edge (1,39). That's chord (1,39).
    # So we could use chord (1,39) instead of (0,2) to get length 39.
    # Similarly, length 38: remove two vertices. E.g., remove 0 and 1: need edge (2,39) or similar.
    # This gets complicated. Let's stick with the current family but see if we can save one edge.
    #
    # Actually, the current solution is already at 59 edges. The theoretical minimum for a pancyclic graph
    # on n vertices is n + floor((n-2)/2) or similar, but for n=40, 59 might be near-optimal.
    # Let's just ensure correctness and return the current construction.
    
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))
    
    for j in range(2, 21):
        edges.add((0, j))
    
    edge_list = sorted(edges)
    
    cycles = []
    for k in range(3, N + 1):
        if k <= 21:
            cycle = list(range(k))
        else:
            start = N - k + 1
            cycle = [0] + list(range(N - 1, start - 1, -1))
        cycles.append(cycle)
    
    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
