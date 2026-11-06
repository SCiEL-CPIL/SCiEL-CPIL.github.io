import time
import random
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    # Strategy: Hamiltonian Cycle + 19 Chords from Vertex 0
    # This minimizes edges to 40 + 19 = 59.
    # Score: 59 - 40 = 19.
    
    edges = set()
    
    # 1. Base Hamiltonian Cycle: (i, (i+1)%N)
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 2. Chords from 0 to k for k in 2..20
    # These cover all cycle lengths 3..39.
    # k=2 -> lengths 3, 39
    # k=3 -> lengths 4, 38
    # ...
    # k=20 -> length 21
    for k in range(2, 21):
        u, v = 0, k
        if u > v:
            u, v = v, u
        edges.add((u, v))
        
    # 3. Generate Cycles
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycle_verts = list(range(N))
        else:
            if L <= 21:
                # Use short path from 0 to L-1
                # Cycle: 0, 1, ..., L-1, 0
                # Requires chord (0, L-1)
                k = L - 1
                cycle_verts = list(range(k + 1))
            else:
                # Use long path from 0 to k via N-1
                # Cycle: 0, N-1, ..., k, 0
                # Requires chord (0, k)
                # Length L = (N - k) + 1 => k = N - L + 1
                k = N - L + 1
                cycle_verts = [0] + list(range(N - 1, k - 1, -1))
        
        cycles.append(cycle_verts)
        
    # Validation
    edge_set = set(tuple(sorted(e)) for e in edges)
    valid = True
    
    # Check edges distinct and valid
    if len(edge_set) != len(edges):
        valid = False
    for u, v in edge_set:
        if not (0 <= u < N and 0 <= v < N and u != v):
            valid = False
            
    # Check cycles
    if len(cycles) != N - 2:
        valid = False
    else:
        for idx, L in enumerate(range(3, N + 1)):
            cycle = cycles[idx]
            if len(cycle) != L:
                valid = False
                break
            if len(set(cycle)) != L:
                valid = False
                break
            for i in range(L):
                u = cycle[i]
                v = cycle[(i + 1) % L]
                if (min(u, v), max(u, v)) not in edge_set:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        return {'edges': [tuple(sorted(e)) for e in edges], 'cycles': cycles}
    else:
        # Fallback: Fan Graph (Score 37)
        fan_edges = set()
        for i in range(N - 1):
            fan_edges.add((i, i + 1))
        for i in range(2, N):
            fan_edges.add((0, i))
            
        fan_cycles = []
        for k in range(3, N + 1):
            fan_cycles.append(list(range(k)))
            
        return {'edges': [tuple(sorted(e)) for e in fan_edges], 'cycles': fan_cycles}
