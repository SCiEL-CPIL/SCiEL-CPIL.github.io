import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Base: Hamiltonian cycle on 0..39
    # Edges: (i, i+1) for i in 0..38, and (39, 0)
    edges = []
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Chords to generate cycles of length 3..40
    # We need to cover lengths 3..39.
    # Chord (0, d) generates cycles of length d+1 and N-d+1.
    # Let k = d+1. Lengths are k and N+2-k.
    # We need to cover all k in 3..39.
    # Pairs: {3, 39}, {4, 38}, ..., {20, 22}, {21, 21}.
    # We need chords for k = 3 to 21.
    
    for k in range(3, 22):
        d = k - 1
        # Edge (0, d)
        # Ensure d is not 0 or 1 (which are already in base cycle)
        # k>=3 => d>=2. So (0,2), (0,3)... are new edges.
        # Check if edge exists. (0, d) is not in base cycle unless d=1 or d=39.
        # d ranges 2..20. So all are new.
        edges.append((0, d))
        
        if time.time() > deadline:
            break

    # Construct cycles
    cycles = []
    
    # Cycle of length 40
    cycles.append(list(range(N)))
    
    # Cycles of length 3..39
    # For each k in 3..21, we have a chord (0, k-1).
    # This creates a cycle of length k: 0, 1, ..., k-1, 0.
    # And a cycle of length N+2-k: 0, k-1, k, ..., N-1, 0.
    
    # We need to output cycles in order of length 3..40.
    # So we generate them and sort or map.
    
    cycle_map = {40: list(range(N))}
    
    for k in range(3, 22):
        # Cycle of length k
        # Vertices: 0, 1, ..., k-1
        c1 = list(range(k))
        cycle_map[k] = c1
        
        # Cycle of length N+2-k
        l2 = N + 2 - k
        if l2 != k: # Avoid duplicate if l2 == k (which happens for k=21, N=40 -> 42-21=21)
            # Vertices: 0, k-1, k, ..., N-1
            # Path: 0 -> k-1 -> k -> ... -> N-1 -> 0
            c2 = [0] + list(range(k-1, N))
            cycle_map[l2] = c2
        else:
            # If l2 == k, we already have c1. 
            # But we need a cycle of length 21. c1 is length 21.
            # The other cycle is also length 21. We just need one.
            pass

    # Build final list in order 3..40
    final_cycles = []
    for length in range(3, N + 1):
        if length in cycle_map:
            final_cycles.append(cycle_map[length])
        else:
            # Fallback (should not happen)
            final_cycles.append(list(range(length)))

    return {'edges': edges, 'cycles': final_cycles}
# EVOLVE-BLOCK-END
