import time
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    # 1. Define Edges
    # Base cycle edges
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords from 0 to k for k in 2..20
    # This covers all cycle lengths 3..39
    # Total chords = 19
    for k in range(2, 21):
        edges.add((0, k))
        
    # Convert to list of tuples for output
    # The task requires "undirected edge pairs". Order within pair doesn't matter, 
    # but let's sort them for consistency.
    edge_list = sorted([(min(u, v), max(u, v)) for u, v in edges])
    
    # 2. Define Cycles
    # We need a list of cycles, one for each length L in 3..40, in that order.
    cycles = []
    
    # Helper to generate a cycle from a start node, going forward mod N
    # But our cycles are specific paths on the base cycle + chord.
    
    for L in range(3, N + 1):
        if L == N:
            # The full cycle
            cycle = list(range(N))
        else:
            if L <= N // 2 + 1:
                # Use chord (0, L-1)
                # Cycle: 0, 1, ..., L-1, 0
                k = L - 1
                cycle = list(range(k + 1))
            else:
                # Use chord (0, N-L+1)
                # Cycle: 0, N-L+1, N-L+2, ..., N-1, 0
                k = N - L + 1
                # Vertices: k, k+1, ..., N-1, 0
                # Note: 0 is the start.
                # Path: 0 -> k -> k+1 -> ... -> N-1 -> 0
                cycle = [0] + list(range(k, N))
        
        cycles.append(cycle)
        
    # Check time
    if time.time() > deadline:
        # This should not happen for static construction
        pass

    return {
        'edges': edge_list,
        'cycles': cycles
    }
