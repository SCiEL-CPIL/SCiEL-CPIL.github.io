"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 40

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    
    # Base: 40-cycle (edges i-(i+1) mod 40)
    cycle_edges = [(i, (i+1) % N) for i in range(N)]
    
    # Strategy: Use chords from vertex 0 to cover all cycle lengths 3..39.
    # Chord (0, k) creates cycles of length k+1 (forward path) and N-k+1 (backward path).
    # We need to cover lengths 3..39.
    # Length 40 is provided by the base cycle.
    # Chords k=2..20 cover:
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=19 -> 20, 22
    # k=20 -> 21, 21
    # This covers all lengths 3..39.
    # Total chords: 19.
    # Total edges: 40 + 19 = 59.
    
    chords = [(0, k) for k in range(2, 21)]
    
    all_edges = cycle_edges + chords
    edge_set = set((min(e), max(e)) for e in all_edges)
    
    # Verify all edges are distinct
    assert len(edge_set) == len(all_edges), f"Duplicate edges: {len(all_edges)} vs {len(edge_set)}"
    
    # Build cycles for each length 3..40
    cycles = []
    
    # Helper: get the path from 0 to k going forward (0,1,2,...,k)
    def forward_path(k):
        return list(range(k+1))
    
    # Helper: get the path from 0 to k going backward (0,39,38,...,k)
    def backward_path(k):
        return [0] + list(range(N-1, k-1, -1))
    
    for length in range(3, N+1):
        if length == N:
            # The full 40-cycle
            cycles.append(list(range(N)))
        elif length == 21:
            # Chord (0,20): forward path 0->1->...->20->0 gives 21 vertices
            cycles.append(forward_path(20))
        else:
            # Find k such that k+1 == length or 41-k == length
            if length <= 20:
                # Use forward path: k = length-1
                k = length - 1
                cyc = forward_path(k)
                cycles.append(cyc)
            else:
                # Use backward path: k = 41-length
                k = 41 - length
                cyc = backward_path(k)
                cycles.append(cyc)
    
    # Verify all cycles are valid
    for i, cyc in enumerate(cycles):
        length = i + 3
        if len(cyc) != length:
            raise ValueError(f"Cycle {length}: length {len(cyc)} != {length}")
        if len(set(cyc)) != length:
            raise ValueError(f"Cycle {length}: duplicate vertices")
        for j in range(len(cyc)):
            u, v = cyc[j], cyc[(j+1) % len(cyc)]
            e = (min(u,v), max(u,v))
            if e not in edge_set:
                raise ValueError(f"Cycle {length}: edge {e} missing")
    
    return {'edges': [list(e) for e in edge_set], 'cycles': cycles}
# EVOLVE-BLOCK-END
