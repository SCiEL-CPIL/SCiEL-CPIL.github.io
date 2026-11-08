"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 120s

    # Generate all weight-4 vectors in R^11
    # A weight-4 vector has exactly 4 non-zero entries, each in {-1, 1}
    # Total count: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors
    
    # Compute Gram matrix
    G = vectors @ vectors.T
    
    # Build adjacency: conflict if G[i,j] > 2
    conflict = G > 2
    np.fill_diagonal(conflict, False)
    
    # Greedy independent set with local search improvement
    in_set = np.zeros(n, dtype=bool)
    remaining = np.ones(n, dtype=bool)
    degrees = conflict.sum(axis=1).astype(np.int32)
    
    # Phase 1: Greedy construction
    while remaining.any():
        if time.time() > deadline:
            break
        
        masked_degrees = np.where(remaining, degrees, np.inf)
        idx = np.argmin(masked_degrees)
        
        if not remaining[idx]:
            break
        
        in_set[idx] = True
        remaining[idx] = False
        
        # Remove neighbors and update degrees
        neighbors = conflict[idx] & remaining
        if neighbors.any():
            degrees[neighbors] -= 1
            remaining[neighbors] = False
    
    # Phase 2: Local search - try to add removed vertices
    if time.time() < deadline - 30:
        in_set_list = np.where(in_set)[0]
        removed = ~in_set
        
        # For each removed vertex, check if it can be added
        # A vertex can be added if it has no conflicts with current in_set
        # We do this in multiple passes to allow chains of additions
        for _ in range(5):
            if time.time() > deadline:
                break
            
            # Find removed vertices that have no conflicts with in_set
            # conflict[v, in_set].any() tells if v conflicts with any in_set vertex
            # We want vertices where conflict[v, in_set] is all False
            has_conflict = conflict[:, in_set].any(axis=1)
            candidates = removed & ~has_conflict
            
            if not candidates.any():
                break
            
            # Add all candidates that don't conflict with each other
            # Use greedy approach on candidates
            cand_indices = np.where(candidates)[0]
            for v_idx in cand_indices:
                if time.time() > deadline:
                    break
                # Check if adding v_idx conflicts with any already-added in this pass
                # Simple approach: just add if no conflict with current in_set
                if not conflict[v_idx, in_set].any():
                    in_set[v_idx] = True
                    removed[v_idx] = False
    
    result = vectors[in_set]
    return result
# EVOLVE-BLOCK-END
