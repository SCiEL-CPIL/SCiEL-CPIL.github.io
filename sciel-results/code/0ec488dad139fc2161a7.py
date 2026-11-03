"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s
    
    # Type A vectors: one coordinate is +/- 2, others 0
    # Norm squared = 4
    type_a = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            type_a.append(tuple(v))
    
    # Type D vectors: four coordinates are +/- 1, others 0
    # Norm squared = 4
    type_d_candidates = []
    # Generate all combinations of 4 indices
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        # Generate all sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(comb, signs):
                v[idx] = s
            type_d_candidates.append(tuple(v))
            
    # We want to select a subset of type_d_candidates such that
    # no two vectors share 3 indices with the same signs.
    # This is equivalent to: no two vectors share a "signed triple".
    
    # Map each signed triple to a set of vector indices (or just track used triples)
    # A signed triple is (i, j, k, s_i, s_j, s_k) where i<j<k and s are the signs at those indices.
    
    used_triples = set()
    selected_type_d = []
    
    # Shuffle candidates to get a random maximal independent set
    # We can do multiple restarts to find a larger set
    import random
    random.seed(42)
    
    best_selected = []
    
    max_restarts = 10
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        candidates = type_d_candidates[:]
        random.shuffle(candidates)
        
        current_used = set()
        current_selected = []
        
        for v in candidates:
            # Extract 4 signed triples
            # Find non-zero indices
            nz_indices = [i for i in range(D) if v[i] != 0]
            # nz_indices has length 4
            triples = []
            # There are 4 triples of 3 indices from 4 indices
            # Indices: 0,1,2,3 -> (0,1,2), (0,1,3), (0,2,3), (1,2,3)
            for triple_idx in itertools.combinations(range(4), 3):
                # Get the actual indices
                actual_indices = tuple(nz_indices[i] for i in triple_idx)
                # Get the signs
                actual_signs = tuple(v[i] for i in actual_indices)
                # Create key
                key = (actual_indices, actual_signs)
                triples.append(key)
            
            # Check if any triple is already used
            conflict = False
            for t in triples:
                if t in current_used:
                    conflict = True
                    break
            
            if not conflict:
                # Add to selected
                current_selected.append(v)
                for t in triples:
                    current_used.add(t)
        
        if len(current_selected) > len(best_selected):
            best_selected = current_selected
            
    # Combine with Type A
    # Type A vectors are compatible with all Type D vectors
    # Check compatibility between Type A vectors themselves
    # Dist between (2,0,..) and (0,2,..) is sqrt(8) > 2. OK.
    # Dist between (2,0,..) and (-2,0,..) is 4 > 2. OK.
    # So all 22 Type A vectors are mutually compatible.
    
    final_vectors = list(type_a) + best_selected
    
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
