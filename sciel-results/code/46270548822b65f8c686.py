"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Strategy: Use disjoint copies of K4.
    # K4 is C5-free (max cycle length in K4 is 3, K4 has no C4 or C5 as induced or subgraph? 
    # Wait, K4 has triangles. Does K4 contain C5? No, C5 needs 5 distinct vertices. K4 has 4.
    # So any disjoint union of K4s is C5-free.
    # Independence number of disjoint union of k K4s is k.
    # We need alpha < 9, so k <= 8.
    # Max n = 8 * 4 = 32.
    
    n = 32
    k = 8
    
    edges = []
    for b in range(k):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    # Verification is implied by construction, but we can add a quick check if time permits
    # For n=32, checking C5 is fast, but unnecessary for this static construction.
    # We trust the property of K4s.
    
    if time.time() > deadline:
        # Fallback to a smaller graph if we somehow took too long (shouldn't happen)
        n = 28
        k = 7
        edges = []
        for b in range(k):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
