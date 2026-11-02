import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate all 4-subsets of range(D)
    # We need a packing of 4-subsets such that any 3 elements are in at most 1 subset.
    # This is equivalent to: any two subsets share at most 2 elements.
    
    all_subsets = list(itertools.combinations(range(D), 4))
    
    # Greedy selection of subsets
    selected_subsets = []
    # To speed up, we can track used triples
    # A triple is a frozenset of 3 elements
    used_triples = set()
    
    # Shuffle to allow different runs? No, deterministic is better for reproducibility, 
    # but greedy order matters. 
    # We can just iterate in lexicographic order.
    
    for subset in all_subsets:
        # Check if this subset conflicts with existing ones
        # Conflict if it shares a triple with any existing subset
        # i.e., if any of its 4 triples is already used
        conflict = False
        for triple in itertools.combinations(subset, 3):
            if frozenset(triple) in used_triples:
                conflict = True
                break
        
        if not conflict:
            selected_subsets.append(subset)
            for triple in itertools.combinations(subset, 3):
                used_triples.add(frozenset(triple))
                
        # Time check
        if time.time() > deadline:
            break
            
    # 2. Generate vectors
    vectors = []
    for subset in selected_subsets:
        # Generate all 16 sign combinations for the 4 indices
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(subset, signs):
                v[idx] = sign
            vectors.append(v)
            
    # 3. Return
    if not vectors:
        # Fallback to seed if something went wrong
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
