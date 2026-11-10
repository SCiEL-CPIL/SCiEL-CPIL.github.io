import numpy as np
from itertools import combinations

N = 40

def construct():
    # Vertices 0 to N-1
    # Base: Cycle C_N
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # We need to cover all cycle lengths L in 3..N-1
    # Chord (0, j) creates cycles of length j+1 and N-j+1
    # We need to select S subset of {2..N-2} such that for all L in 3..N-1,
    # L-1 in S or N-L+1 in S
    # This is equivalent to picking one from each pair {j, N-j} for j in 2..N-2
    
    # For N=40:
    # Pairs are {j, 40-j}
    # j=2 -> 38
    # j=3 -> 37
    # ...
    # j=19 -> 21
    # j=20 -> 20
    
    # The current program picks j for j in 2..20
    # This gives 19 chords
    # Edges = 40 (cycle) + 19 (chords) = 59
    # Score = 59 - 40 = 19
    
    # Let's see if we can do better
    # The pairs are {2,38}, {3,37}, {4,36}, {5,35}, {6,34}, {7,33}, {8,32}, {9,31}, {10,30}, {11,29}, {12,28}, {13,27}, {14,26}, {15,25}, {16,24}, {17,23}, {18,22}, {19,21}, {20,20}
    # That's 19 pairs
    # We must pick at least one from each pair
    # So minimum 19 chords
    # Thus minimum edges = 59
    # Minimum score = 19
    
    # The current program already achieves this
    # But let's verify the cycle generation is correct
    
    S = set()
    for j in range(2, N // 2 + 1):
        S.add(j)
        
    for j in S:
        edges.add((0, j))
        
    edge_list = list(edges)
    
    # Generate cycles
    # cycles[k] should be a cycle of length k+3
    # Lengths needed: 3, 4, ..., 40
    
    cycles = [None] * (N - 2) # Index 0 is len 3, Index N-3 is len N
    
    # Length N is the base cycle
    cycles[N - 3] = list(range(N))
    
    # For each chord j in S, we have two cycles
    # Cycle 1: 0, 1, ..., j, 0 (Length j+1)
    # Cycle 2: 0, j, j+1, ..., N-1, 0 (Length N-j+1)
    
    for j in S:
        len1 = j + 1
        len2 = N - j + 1
        
        # Cycle 1
        c1 = list(range(0, j + 1))
        
        # Cycle 2
        c2 = [0] + list(range(j, N))
        
        # Assign to correct slots
        if len1 <= N - 1:
            cycles[len1 - 3] = c1
        if len2 <= N - 1:
            cycles[len2 - 3] = c2
            
    # Verify all cycles are filled
    for i, c in enumerate(cycles):
        if c is None:
            raise ValueError(f"Cycle of length {i+3} not generated")
    
    # Verify all cycles are valid
    for c in cycles:
        if len(set(c)) != len(c):
            raise ValueError(f"Cycle has duplicate vertices: {c}")
        for i in range(len(c)):
            u = c[i]
            v = c[(i+1) % len(c)]
            if (u, v) not in edges and (v, u) not in edges:
                raise ValueError(f"Edge ({u}, {v}) not in graph for cycle {c}")
    
    return {'edges': edge_list, 'cycles': cycles}
