import numpy as np
import itertools
import random

D = 11

def construct():
    # Generate all candidates with norm 2 (4 ones)
    supports = list(itertools.combinations(range(D), 4))
    candidates = []
    for sup in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, s in zip(sup, signs):
                v[i] = s
            candidates.append(v)
            
    # Total candidates: 330 * 16 = 5280
    
    # Precompute triples for each support
    # support is a tuple of 4 indices
    # triples are 4 subsets of size 3
    support_triples = {}
    for sup in supports:
        triples = list(itertools.combinations(sup, 3))
        support_triples[sup] = triples

    # Slot usage: (triple_tuple, sign_tuple) -> count
    # We want to select a set of vectors
    # Greedy with random restarts
    
    best_vectors = []
    
    # To speed up, we can just try a few random orders
    for restart in range(10):
        random.shuffle(candidates)
        used_slots = set()
        current_vectors = []
        
        for v in candidates:
            # Find the support for this vector?
            # We can reconstruct support from v
            sup = tuple(i for i in range(D) if v[i] != 0)
            triples = support_triples[sup]
            
            valid = True
            for tri in triples:
                # Get signs on tri
                tri_signs = tuple(v[i] for i in tri)
                slot_key = (tri, tri_signs)
                if slot_key in used_slots:
                    valid = False
                    break
            
            if valid:
                current_vectors.append(v)
                for tri in triples:
                    tri_signs = tuple(v[i] for i in tri)
                    used_slots.add((tri, tri_signs))
                    
        if len(current_vectors) > len(best_vectors):
            best_vectors = current_vectors
            
    return np.array(best_vectors, dtype=np.int64)
