import numpy as np
import itertools
import random

D = 11

def construct():
    # Generate all 4-subsets of indices 0..10
    all_subsets = list(itertools.combinations(range(D), 4))
    
    best_blocks = []
    
    # Try multiple random orderings to find a large packing
    # A packing is a set of subsets where any two intersect in at most 2 elements.
    # This is equivalent to no 3-subset being contained in more than one block.
    
    for _ in range(500):
        random.shuffle(all_subsets)
        current_blocks = []
        covered_3s = set()
        
        for block in all_subsets:
            block_3s = list(itertools.combinations(block, 3))
            valid = True
            for t in block_3s:
                if t in covered_3s:
                    valid = False
                    break
            
            if valid:
                current_blocks.append(block)
                for t in block_3s:
                    covered_3s.add(t)
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            if len(best_blocks) >= 41:
                break
    
    # Construct vectors
    vectors = []
    
    # All 16 sign combinations for 4 dimensions
    all_signs = list(itertools.product([1, -1], repeat=4))
    
    for block in best_blocks:
        for signs in all_signs:
            v = [0] * D
            for idx, sign in zip(block, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
