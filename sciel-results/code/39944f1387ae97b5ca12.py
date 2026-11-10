"""
Constructs a sparse pancyclic graph on 40 vertices.
Target: Minimize (edge_count - 40).
Strategy:
1. Base Hamiltonian cycle 0-1-...-39-0 (40 edges).
2. Add chords (0, k) for k in 2..20 (19 edges).
Total edges: 59. Score: 19.
Cycles:
- L=40: Base cycle.
- L in 3..20: Path 0->1->...->(L-1) + Chord (L-1, 0).
- L=21: Path 0->1->...->20 + Chord (20, 0).
- L in 22..39: Path 0->39->...->k + Chord (k, 0) where k = 41-L.
"""

import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    edges = []
    # Base Hamiltonian cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords from 0 to k for k in 2..20
    for k in range(2, 21):
        edges.append((0, k))
        
    edges_list = edges
    
    cycles = []
    # Lengths 3 to 40
    for L in range(3, N + 1):
        if L == 40:
            # Full Hamiltonian cycle
            cycle = list(range(N))
        elif 3 <= L <= 21:
            # Use chord (0, L-1)
            # Cycle: 0, 1, ..., L-1
            cycle = list(range(L))
        else:
            # L in 22..39
            # Use chord (0, k) where k = 41 - L
            k = 41 - L
            # Cycle: 0, 39, 38, ..., k
            # Vertices: 0, then 39 down to k
            cycle = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cycle)
        
    return {'edges': edges_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
