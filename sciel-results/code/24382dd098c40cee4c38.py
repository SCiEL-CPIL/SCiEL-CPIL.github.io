"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Strategy: Start with a Hamiltonian cycle (40 edges) + chords from vertex 0
    # We need cycles of every length 3..40.
    # A Hamiltonian cycle gives length 40.
    # Chords from vertex 0 to other vertices create shorter cycles.
    # If we have a cycle 0-1-2-...-39-0, adding chord (0, k) creates a cycle of length k+1 (0-1-...-k-0) and another of length 40-(k-1) = 41-k.
    # So a single chord (0, k) gives cycles of length k+1 and 41-k.
    # We need lengths 3, 4, ..., 39 (40 is already there).
    # Length 3: chord (0,2) -> cycle 0-1-2-0 (len 3) and 0-2-3-...-39-0 (len 39).
    # Length 4: chord (0,3) -> cycle 0-1-2-3-0 (len 4) and 0-3-4-...-39-0 (len 38).
    # ...
    # Length 20: chord (0,19) -> cycle len 20 and len 21.
    # Length 21: chord (0,20) -> cycle len 21 and len 20.
    # So we need chords to cover lengths 3..39.
    # Chord (0, k) covers k+1 and 41-k.
    # We need to cover {3, 4, ..., 39}.
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # k=4 -> 5, 37
    # ...
    # k=19 -> 20, 21
    # k=20 -> 21, 20
    # So chords (0,2), (0,3), ..., (0,19) cover:
    # 3,39; 4,38; 5,37; ...; 20,21.
    # This covers all lengths 3..39.
    # Number of chords: 18 (from k=2 to k=19).
    # Total edges: 40 (cycle) + 18 (chords) = 58 edges.
    # Score = 58 - 40 = 18.
    
    # Can we do better?
    # The minimum number of edges for a pancyclic graph on n vertices is n (Hamiltonian cycle). But we need more.
    # A known result is that n + floor(n/2) - 2 or similar might be tight?
    # For n=40, 58 edges gives score 18.
    # Is it possible to have fewer?
    # If we use a different base graph, maybe a "fan" or "wheel"?
    # Wheel graph W_n has n+1 vertices, n+1 edges (cycle + spokes). Here N=40.
    # Let's stick with the Hamiltonian cycle + chords strategy first, as it's simple and provably works.
    # We can try to optimize the chord placement to see if we can share chords for multiple cycle lengths?
    # No, each chord (0,k) creates exactly two cycles involving the path from 0 to k along the Hamiltonian cycle.
    # Other cycles might exist, but we only need one of each length.
    # So 18 chords seems minimal for this specific construction.
    # Total edges 58. Score 18.
    
    # Let's verify if we can reduce edges further.
    # What if we don't use a single Hamiltonian cycle?
    # Maybe two cycles connected by a few edges?
    # If we have two 20-cycles, that's 40 edges. Connect them with 2 edges?
    # This might allow cycles of various lengths.
    # But constructing all lengths 3..40 is tricky.
    # The Hamiltonian + chords approach is very robust.
    
    # Let's implement the Hamiltonian + chords approach.
    # Vertices 0..39.
    # Cycle: 0-1-2-...-39-0.
    # Chords: (0,2), (0,3), ..., (0,19).
    
    edges = set()
    # Hamiltonian cycle
    for i in range(N):
        edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
    
    # Chords
    chords = []
    for k in range(2, 20):  # k=2..19
        edges.add((0, k))
        chords.append(k)
    
    # Now we need to generate the cycles.
    # For each length L in 3..40:
    # If L == 40: the Hamiltonian cycle.
    # Else: find a chord (0, k) such that L == k+1 or L == 41-k.
    # If L == k+1, the cycle is 0-1-...-k-0.
    # If L == 41-k, the cycle is 0-k-(k+1)-...-39-0.
    
    cycles = []
    for L in range(3, N+1):
        if L == N:
            cycles.append(list(range(N)))
        else:
            found = False
            for k in range(2, 20):
                if L == k + 1:
                    # Cycle 0-1-...-k-0
                    cyc = list(range(0, k+1))
                    cycles.append(cyc)
                    found = True
                    break
                elif L == 41 - k:
                    # Cycle 0-k-(k+1)-...-39-0
                    cyc = [0] + list(range(k, N))
                    cycles.append(cyc)
                    found = True
                    break
            if not found:
                # This should not happen
                raise ValueError(f"Could not find cycle for length {L}")
    
    result = {'edges': list(edges), 'cycles': cycles}
    
    # Verify
    edge_set = set()
    for e in result['edges']:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    
    for i, cyc in enumerate(result['cycles']):
        L = i + 3
        if len(cyc) != L:
            raise ValueError(f"Cycle length mismatch at index {i}")
        if len(set(cyc)) != L:
            raise ValueError(f"Cycle has duplicate vertices at index {i}")
        for j in range(L):
            u = cyc[j]
            v = cyc[(j+1)%L]
            if (min(u,v), max(u,v)) not in edge_set:
                raise ValueError(f"Missing edge in cycle at index {i}")
    
    return result
# EVOLVE-BLOCK-END
