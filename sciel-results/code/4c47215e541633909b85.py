"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sparse pancyclic graph on N=40 vertices.
    Strategy: Hamiltonian cycle + chords from vertex 0.
    Minimize edge count - 40.
    """
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Base: Hamiltonian cycle 0-1-2-...-39-0 (40 edges)
    # We need cycles of every length 3..40.
    
    # The Hamiltonian cycle gives us a cycle of length 40.
    # We need cycles of lengths 3, 4, ..., 39.
    
    # Strategy: Use vertex 0 as a hub for chords.
    # A chord (0, k) creates a cycle of length k (path 0-1-...-k-0) 
    # and a cycle of length 40-k (path 0-39-...-k-0).
    # Wait, the cycle using chord (0,k) and the path along the Hamiltonian cycle:
    # Path 0-1-...-k has length k edges, plus chord (0,k) makes a cycle of length k+1 vertices?
    # No, cycle length is number of edges.
    # Path 0-1-...-k is k edges. Adding chord (0,k) makes a cycle with k+1 edges? No.
    # Vertices: 0, 1, ..., k. Edges: (0,1), (1,2), ..., (k-1,k), (k,0).
    # Number of edges = k. Number of vertices = k+1.
    # Cycle length is usually defined as number of vertices (or edges, same for cycles).
    # So chord (0,k) creates a cycle of length k+1 and a cycle of length 40-(k-1) = 41-k?
    # Let's re-verify.
    # Hamiltonian cycle: 0-1-2-...-39-0.
    # Chord (0, k).
    # Cycle 1: 0-1-...-k-0. Vertices: 0..k. Count: k+1.
    # Cycle 2: 0-k-(k-1)-...-1-0? No, the other way around the Hamiltonian cycle.
    # Path from k to 0 going the other way: k-(k+1)-...-39-0.
    # Vertices: k, k+1, ..., 39, 0. Count: 40 - k + 1 = 41 - k.
    # So chord (0, k) gives cycles of length k+1 and 41-k.
    
    # We need cycles of length 3 to 40.
    # Length 40 is already there (Hamiltonian).
    # We need 3, 4, ..., 39.
    
    # Let's pick chords to cover all required lengths.
    # If we pick chord (0, k), we get lengths k+1 and 41-k.
    # We want to cover {3, 4, ..., 39}.
    
    # Let's try to minimize the number of chords.
    # Each chord covers 2 lengths (unless k+1 == 41-k, i.e., 2k=40, k=20, then length 21).
    # We need 37 lengths (3..39).
    # Minimum chords: ceil(37/2) = 19 chords.
    # Total edges: 40 (Hamiltonian) + 19 (chords) = 59.
    # Score: 59 - 40 = 19.
    
    # Let's verify if 19 chords can cover all 37 lengths.
    # Lengths available from chord k: {k+1, 41-k}.
    # We need to select a set of k's such that the union of {k+1, 41-k} for k in set covers {3..39}.
    
    # Let's try to construct such a set.
    # Note that k ranges from 1 to 39 (since 0 is the hub, and 0-0 is not an edge).
    # Actually, k can be 1..39.
    # If k=1: lengths 2, 40. (2 is not needed, 40 is already there).
    # If k=2: lengths 3, 39.
    # If k=3: lengths 4, 38.
    # ...
    # If k=19: lengths 20, 22.
    # If k=20: lengths 21, 21.
    # If k=21: lengths 22, 20.
    # ...
    
    # We need to cover 3, 4, ..., 39.
    # Let's pair them up.
    # 3 and 39: k=2.
    # 4 and 38: k=3.
    # 5 and 37: k=4.
    # ...
    # 20 and 22: k=19.
    # 21: k=20.
    
    # So the set of k's is {2, 3, 4, ..., 19, 20}.
    # That's 19 chords.
    # Let's check the coverage:
    # k=2: 3, 39
    # k=3: 4, 38
    # k=4: 5, 37
    # ...
    # k=19: 20, 22
    # k=20: 21
    # Union: {3, 39, 4, 38, 5, 37, ..., 20, 22, 21} = {3, 4, ..., 39}.
    # Yes, this works perfectly.
    
    # So we use chords (0, k) for k in {2, 3, ..., 20}.
    # Total chords: 19.
    # Total edges: 40 + 19 = 59.
    # Score: 19.
    
    # Now we need to generate the cycles explicitly.
    # For each length L in 3..40, we need to provide a cycle of that length.
    # For L=40: the Hamiltonian cycle 0-1-...-39-0.
    # For L in 3..39:
    #   If L is odd, say L=2j+1. Then L=41-k => k=41-L.
    #   If L is even, say L=2j. Then L=k+1 => k=L-1.
    #   Wait, each chord k gives two lengths: k+1 and 41-k.
    #   For a given L, we can pick either chord.
    #   Let's just pick one.
    #   If L <= 21, use k = L-1. (Since k+1 = L => k = L-1).
    #   If L > 21, use k = 41-L. (Since 41-k = L => k = 41-L).
    #   Let's check:
    #   L=3: k=2. Cycle 0-1-2-0. Length 3. Correct.
    #   L=21: k=20. Cycle 0-1-...-20-0. Length 21. Correct.
    #   L=22: k=19. Cycle 0-19-18-...-1-0? No, the other way.
    #   Wait, for k=19, the two cycles are:
    #   1. 0-1-...-19-0 (length 20)
    #   2. 0-19-20-...-39-0 (length 22)
    #   So for L=22, we use the cycle 0-19-20-...-39-0.
    #   Let's formalize the cycle generation.
    
    edges = []
    # Hamiltonian cycle edges
    for i in range(N):
        edges.append((i, (i+1) % N))
    
    # Chords
    chords = []
    for k in range(2, 21):  # k = 2, 3, ..., 20
        edges.append((0, k))
        chords.append(k)
    
    # Remove duplicates just in case (though there shouldn't be any)
    edges = list(set(edges))
    # Ensure undirected pairs are consistent (e.g., (u,v) with u<v)
    edges = [(min(u,v), max(u,v)) for u,v in edges]
    edges = sorted(set(edges))
    
    cycles = []
    for L in range(3, N+1):
        if L == N:
            # Hamiltonian cycle
            cycle = list(range(N))
        else:
            if L <= 21:
                k = L - 1
                # Cycle: 0-1-...-k-0
                cycle = list(range(k+1))
            else:
                k = 41 - L
                # Cycle: 0-k-(k+1)-...-39-0
                # Vertices: 0, k, k+1, ..., 39
                cycle = [0] + list(range(k, N))
        cycles.append(cycle)
    
    # Verify cycles
    edge_set = set(edges)
    for cycle in cycles:
        L = len(cycle)
        for i in range(L):
            u = cycle[i]
            v = cycle[(i+1) % L]
            if (min(u,v), max(u,v)) not in edge_set:
                # This should not happen
                pass
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
