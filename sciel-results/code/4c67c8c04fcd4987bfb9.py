import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Möbius-ladder backbone: two disjoint 20-cycles connected by 20 rungs
    # A-cycle: 0-1-2-...-19-0
    # B-cycle: 20-21-22-...-39-20
    # Rungs: (i, 20+i) for i=0..19
    
    edges = set()
    
    # A-cycle: 0-1-2-...-19-0
    for i in range(20):
        a, b = i, (i + 1) % 20
        edges.add((min(a, b), max(a, b)))
    
    # B-cycle: 20-21-22-...-39-20
    for i in range(20):
        a, b = 20 + i, 20 + (i + 1) % 20
        edges.add((min(a, b), max(a, b)))
    
    # Rungs: (i, 20+i) for i=0..19
    for i in range(20):
        edges.add((i, 20 + i))
    
    # Additional chords to cover remaining cycle lengths
    # We need to create cycles of lengths 3-39
    # The rungs already give us some cycles, but we need more
    
    # Strategy: Use chords from vertex 0 to create short cycles
    # (0,1) is already in A-cycle
    # (0,20) is a rung
    # Add chords (0,k) for k=2..19 to create cycles
    
    # Actually, let me use a simpler approach:
    # Keep the original ring + chords approach, but verify it's optimal
    
    # For now, let me just return the original solution
    # The Möbius ladder is more complex than needed
    
    edges = set()
    
    # Ring: 0-1-2-...-39-0
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))
    
    # Chords from vertex 0 to vertices 2..20
    for k in range(2, 21):
        edges.add((0, k))
    
    # Build cycle certificates for each length 3..40
    cycles = {}
    
    # Length 40: the full ring
    cycles[40] = list(range(N))
    
    # For each chord k, assign cycles to L1=k+1 and L2=41-k
    for k in range(2, 21):
        l1 = k + 1
        l2 = 41 - k
        
        if l1 not in cycles:
            # Forward path: 0, 1, 2, ..., k
            cycles[l1] = list(range(k + 1))
        
        if l2 not in cycles:
            # Backward path: 0, 39, 38, ..., k
            cycles[l2] = [0] + list(range(N - 1, k - 1, -1))
    
    # Assemble final list in order: length 3, 4, ..., 40
    final_cycles = []
    for L in range(3, 41):
        final_cycles.append(cycles[L])
    
    # Convert edges to sorted list of tuples
    edge_list = sorted(edges)
    
    return {'edges': edge_list, 'cycles': final_cycles}


if __name__ == '__main__':
    res = construct()
    print(f"Edges: {len(res['edges'])}")
    print(f"Cycles: {len(res['cycles'])}")
    # Quick validation
    edge_set = {frozenset(e) for e in res['edges']}
    for c in res['cycles']:
        assert len(c) == len(set(c)), f"Duplicate vertex in cycle of length {len(c)}"
        for i in range(len(c)):
            u, v = c[i], c[(i + 1) % len(c)]
            assert frozenset((u, v)) in edge_set, f"Missing edge ({u},{v}) in cycle of length {len(c)}"
    print("All cycles valid.")
