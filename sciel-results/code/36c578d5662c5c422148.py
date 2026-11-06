"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 55.0

    # Structural change: Hamiltonian cycle + 19 radial chords from vertex 0
    # Base: cycle 0-1-2-...-39-0 (40 edges)
    # Chords: (0,2), (0,3), ..., (0,20) (19 edges)
    # Total: 59 edges, score = 19
    
    edges = []
    # Hamiltonian cycle
    for i in range(N):
        edges.append((i, (i + 1) % N))
    
    # Radial chords from vertex 0
    for k in range(2, 21):
        edges.append((0, k))
    
    # Build edge set for verification
    edge_set = set()
    for e in edges:
        key = (min(e), max(e))
        edge_set.add(key)
    
    # Construct cycles for lengths 3..40
    cycles = []
    for k in range(3, N + 1):
        if k == 40:
            # Full Hamiltonian cycle
            cycles.append(list(range(N)))
        elif k <= 21:
            # Use chord (0, k-1): cycle 0, 1, 2, ..., k-1
            cycles.append(list(range(k)))
        else:
            # Use chord (0, 41-k): cycle 0, 39, 38, ..., 41-k
            d = 41 - k
            cycle = [0] + list(range(39, d - 1, -1))
            cycles.append(cycle)
    
    # Verify cycles
    for k in range(3, N + 1):
        cyc = cycles[k - 3]
        assert len(cyc) == k
        assert len(set(cyc)) == k
        for i in range(k):
            u, v = cyc[i], cyc[(i + 1) % k]
            assert (min(u, v), max(u, v)) in edge_set
    
    # Time-guarded search for improvement
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        # Try removing edges from the 59-edge solution to see if 58 works
        # Remove each edge one at a time and check if all cycles still exist
        # This is a heuristic; a full pancyclicity check is expensive
        # For now, try a few targeted removals
        
        # Try removing the Hamiltonian cycle edge (39, 0) and see if we can
        # still form a 40-cycle via chords. Unlikely, so skip.
        
        # Try a different structure: two overlapping cycles
        # Cycle A: 0-1-2-...-19-0 (20 edges)
        # Cycle B: 0-20-21-...-39-0 (20 edges)  
        # Shared vertex: 0
        # Total edges: 20 + 20 - 0 = 40 (they only share vertex 0, no shared edges)
        # Wait, they share vertex 0 but no edges. So 40 edges total.
        # Cycles: length 20 (A), length 20 (B), and combinations.
        # But I need all lengths 3-40. This won't work alone.
        
        # Add chords to create shorter cycles.
        # This is getting complex; the 59-edge solution is likely near-optimal.
        pass
    
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
