import itertools

N = 40

def construct():
    # Start with a Hamiltonian cycle (40 edges)
    base_edges = set()
    for i in range(N):
        base_edges.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))
    
    # We need to add chords to create cycles of all lengths 3..39.
    # The Hamiltonian cycle provides C40.
    # We need C3, C4, ..., C39.
    
    # Strategy: Use a "fan" from vertex 0 to create small cycles, 
    # and verify that larger cycles can be formed using the base cycle + chords.
    
    # Let's try to find a minimal set of chords.
    # A known construction for pancyclic graphs uses O(n log n) or O(n) extra edges.
    # For n=40, we can likely do better than 19 chords.
    
    # Let's use a greedy or heuristic approach to find a small set of chords
    # such that for every k in 3..39, there is a cycle of length k.
    
    # Actually, the seed uses 19 chords.
    # Let's try to reduce this.
    
    # Key insight: If we have a Hamiltonian cycle, a chord (u, v) creates a cycle of length
    # equal to the distance between u and v along the Hamiltonian cycle (in one direction) + 1.
    # Specifically, if the vertices are 0..n-1 in order, chord (i, j) with i < j creates
    # a cycle of length (j - i + 1) and another of length (n - (j - i) + 1).
    
    # So, if we add chord (i, j), we get cycles of lengths L1 = j - i + 1 and L2 = n - (j - i) + 1.
    # We need to cover all lengths 3..n-1.
    # Note that L1 + L2 = n + 2.
    
    # Let's see if we can cover all lengths with fewer chords.
    # Lengths needed: 3, 4, ..., 39.
    # Chord (i, j) covers {j - i + 1, n - (j - i) + 1}.
    # Let d = j - i. Then lengths are d + 1 and n - d + 1.
    # d ranges from 1 to n-1.
    # If d=1, lengths are 2 (invalid, simple cycle must be >=3) and n. (n is already covered by base).
    # If d=2, lengths are 3 and n-1.
    # If d=3, lengths are 4 and n-2.
    # ...
    # If d=k, lengths are k+1 and n-k+1.
    
    # We need to cover {3, 4, ..., n-1}.
    # Note that n is covered by base.
    # So we need to cover 3..n-1.
    # Each chord covers two lengths (unless d+1 == n-d+1, i.e., 2d = n, d=n/2).
    # For n=40, d=20 covers length 21 and 21 (just one length).
    # So we need to cover 3..39.
    # Pairs (d+1, n-d+1):
    # d=2 -> (3, 39)
    # d=3 -> (4, 38)
    # d=4 -> (5, 37)
    # ...
    # d=19 -> (20, 22)
    # d=20 -> (21, 21)
    # d=21 -> (22, 20) -- same as d=19
    # So we only need d from 2 to 20.
    # d=2 covers 3, 39
    # d=3 covers 4, 38
    # ...
    # d=19 covers 20, 22
    # d=20 covers 21
    
    # Total chords needed: 19 (d=2..20).
    # This matches the seed's 19 chords!
    # The seed uses chords from 0 to even vertices 2, 4, ..., 38.
    # Let's check the distances for these chords.
    # Chord (0, 2): d=2. Covers 3, 39.
    # Chord (0, 4): d=4. Covers 5, 37.
    # Chord (0, 6): d=6. Covers 7, 35.
    # ...
    # Chord (0, 38): d=38. But distance is min(38, 40-38)=2. Wait.
    # The distance d is the number of edges along the Hamiltonian cycle.
    # For (0, v), the distances are v and n-v.
    # The chord creates cycles of length v+1 and n-v+1.
    # Seed uses v in {2, 4, ..., 38}.
    # v=2: lengths 3, 39.
    # v=4: lengths 5, 37.
    # v=6: lengths 7, 35.
    # ...
    # v=38: lengths 39, 3.
    # So the seed covers:
    # 3, 39 (from v=2, v=38)
    # 5, 37 (from v=4, v=36)
    # 7, 35 (from v=6, v=34)
    # ...
    # It misses even lengths! 4, 6, 8, ..., 38.
    # Wait, the seed program says:
    # "1. Odd cycles L = 2k+1 via chord (0, 2k)" -> This is correct for odd lengths.
    # "2. Even cycles L = 2m via chords (0, 2) and (0, 2m)" -> Let's check this.
    # For even length L=2m, the cycle is 0-2-3-...-2m-0.
    # Edges: (0,2), (2,3), ..., (2m-1, 2m), (2m, 0).
    # This requires chords (0,2) and (0,2m).
    # So for L=4 (m=2), we need (0,2) and (0,4).
    # For L=6 (m=3), we need (0,2) and (0,6).
    # ...
    # For L=38 (m=19), we need (0,2) and (0,38).
    # So the seed uses chords (0,2), (0,4), ..., (0,38).
    # Total chords: 19.
    # This covers all even lengths 4..38 and odd lengths 3..39.
    # Is it possible to do better?
    
    # To cover all lengths 3..39, we need to cover 37 distinct lengths.
    # Each chord covers at most 2 lengths.
    # So we need at least ceil(37/2) = 19 chords.
    # Thus, 19 chords is the theoretical minimum for this "chord on Hamiltonian cycle" approach.
    
    # Can we do better with a different structure?
    # Maybe not a single Hamiltonian cycle + chords?
    # But the problem asks to minimize edge count.
    # Base edges: 40.
    # Chords: 19.
    # Total edges: 59.
    # Score: 59 - 40 = 19.
    
    # Let's verify if 19 is indeed the minimum.
    # A pancyclic graph on n vertices must have at least n + n/2 - 2 edges?
    # No, that's not a standard bound.
    # But for n=40, 19 chords seems very tight.
    
    # Let's try to construct the cycles explicitly to ensure they are valid.
    
    edges = set(base_edges)
    chords = []
    for v in range(2, N, 2):
        edges.add((0, v))
        chords.append(v)
        
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycle = list(range(N))
        elif k % 2 == 1:
            # Odd length k.
            # k = 2m + 1.
            # Chord (0, k-1) = (0, 2m).
            # Cycle: 0-1-...-(k-1)-0.
            # Vertices: 0, 1, ..., k-1.
            # Edges: (0,1), (1,2), ..., (k-2, k-1), (k-1, 0).
            # (k-1, 0) is a chord if k-1 > 1.
            # For k=3, chord (0,2).
            # For k=5, chord (0,4).
            # This works.
            cycle = list(range(k))
        else:
            # Even length k.
            # k = 2m.
            # Cycle: 0-2-3-...-k-0.
            # Vertices: 0, 2, 3, ..., k.
            # Edges: (0,2), (2,3), ..., (k-1, k), (k, 0).
            # (0,2) is a chord.
            # (k,0) is a chord.
            # This works.
            cycle = [0] + list(range(2, k + 1))
        cycles.append(cycle)
        
    return {'edges': sorted(list(edges)), 'cycles': cycles}

BET: {mechanism: "The current solution uses 19 chords, which is the theoretical minimum for the Hamiltonian cycle + chords approach (ceil(37/2)=19). It is highly likely optimal or near-optimal.", q: 0.1, delta: [0, 0]}
