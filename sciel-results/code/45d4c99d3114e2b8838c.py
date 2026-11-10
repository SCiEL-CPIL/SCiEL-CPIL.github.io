import numpy as np

def construct():
    N = 40
    edges = []
    cycles = []

    # Base Hamiltonian cycle C40: (0,1), (1,2), ..., (39,0)
    for i in range(N):
        edges.append((i, (i + 1) % N))

    # Add 19 chords from vertex 0 to vertices 2..20
    # This creates a "fan" structure that enables cycles of all lengths
    for k in range(2, N // 2 + 1):
        edges.append((0, k))

    # Build cycles of every length 3..40
    for L in range(3, N + 1):
        if L == N:
            # Full Hamiltonian cycle
            cyc = list(range(N))
        elif L <= N // 2 + 1:
            # Short cycles: use path 0,1,2,...,(L-1) and chord (L-1, 0)
            # For L=3: 0,1,2,0 (chord (0,2) exists)
            # For L=4: 0,1,2,3,0 (chord (0,3) exists)
            # ...
            # For L=21: 0,1,2,...,20,0 (chord (0,20) exists)
            cyc = list(range(L))
        else:
            # Long cycles: use path 0,39,38,...,(N-L+1),0
            # For L=22: 0,39,38,...,18,0 (19 vertices from 39 down to 18, plus 0 = 20... wait)
            # Let me recalculate:
            # We need L vertices. Path goes 0, 39, 38, ..., m, 0
            # Number of vertices: 1 + (39 - m + 1) = 40 - m
            # We need 40 - m = L, so m = 40 - L
            # For L=22: m = 18, vertices: 0, 39, 38, ..., 18 (that's 39-18+1=22 vertices including 0? No)
            # Vertices are 0, 39, 38, ..., 18. Count: 1 + (39-18+1) = 1 + 22 = 23. That's too many.
            # Let me recount: from 39 down to 18 inclusive is 39-18+1 = 22 vertices, plus 0 makes 23.
            # We need exactly L=22 vertices.
            # So we need 0, 39, 38, ..., m where 1 + (39-m+1) = L => 40-m = L => m = 40-L
            # For L=22: m=18, count = 1+(39-18+1) = 23. Still 23.
            # Wait: 39,38,...,18 has 39-18+1 = 22 elements. Plus 0 = 23 total. But we need 22.
            # The issue: the cycle is 0 -> 39 -> 38 -> ... -> m -> 0
            # Edges: (0,39), (39,38), ..., (m+1,m), (m,0)
            # (0,39) is in C40. (39,38),...,(m+1,m) are in C40. (m,0) must be a chord.
            # m = 40-L. For L=22, m=18. Chord (0,18) exists since 18 is in 2..20.
            # Vertices: 0, 39, 38, ..., 18. That's 1 + (39-18+1) = 23 vertices. But L=22!
            # I'm miscounting. list(range(39, 18-1, -1)) = [39,38,...,19,18] which has 22 elements.
            # Plus [0] makes 23. But we need 22.
            # The problem: for L=22, we need 22 distinct vertices in the cycle.
            # 0 + [39,38,...,18] = 23 vertices. Off by one.
            # Let me reconsider: we need L vertices total including 0.
            # So we need L-1 vertices from the tail: 39, 38, ..., m
            # L-1 = 39-m+1 = 40-m => m = 40-(L-1) = 41-L
            # For L=22: m = 19. Vertices: 0, 39, 38, ..., 19. Count: 1+(39-19+1)=1+21=22. Correct!
            # Chord needed: (0, 19). 19 is in 2..20. Exists!
            m = 41 - L
            cyc = [0] + list(range(39, m - 1, -1))
        cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}
