import numpy as np
import itertools
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate all 4-subsets of indices 0..10
    all_subsets = list(itertools.combinations(range(D), 4))
    
    best_blocks = []
    
    # Try multiple random orderings to find a large packing
    # A packing is a set of subsets where any two intersect in at most 2 elements.
    # This is equivalent to no 3-subset being contained in more than one block.
    
    for _ in range(200):
        random.shuffle(all_subsets)
        current_blocks = []
        # To speed up, we can track used 3-subsets?
        # Total 3-subsets is C(11,3) = 165.
        # We can maintain a set of "forbidden" 3-subsets?
        # No, if a 3-subset is in block A, it cannot be in block B.
        # So we can just maintain a set of all 3-subsets that have been "covered".
        # If a new candidate block contains any covered 3-subset, it's invalid.
        
        covered_3s = set()
        
        for block in all_subsets:
            # Generate all 3-subsets of this block
            block_3s = list(itertools.combinations(block, 3))
            
            # Check if any are already covered
            valid = True
            for t in block_3s:
                if t in covered_3s:
                    valid = False
                    break
            
            if valid:
                current_blocks.append(block)
                for t in block_3s:
                    covered_3s.add(t)
                
                # Early stop if we can't beat the best?
                # Max possible is 33.
                # If len(current_blocks) + (len(all_subsets) - index) < len(best_blocks): break?
                # Optimization: if current + remaining < best, break.
                # But we don't know remaining easily without index.
                pass
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            if len(best_blocks) >= 32: # 32*8 = 256, good target
                # We can stop if we hit a high number, but let's try a bit more
                # 33 is impossible, so 32 is likely max or near max.
                pass

    # Construct vectors
    vectors = []
    
    # Even parity sign vectors for 4 dimensions
    # 1: 1, 1, 1, 1 (0 minus ones)
    # 2: 1, 1, -1, -1 (2 minus ones)
    # etc.
    even_signs = []
    for signs in itertools.product([1, -1], repeat=4):
        # Count -1s
        if signs.count(-1) % 2 == 0:
            even_signs.append(signs)
    
    # There are 8 such vectors.
    
    for block in best_blocks:
        # Map block indices to 0,1,2,3 for sign assignment
        # We just need to place the signs in the correct positions
        for signs in even_signs:
            v = [0] * D
            for idx, sign in zip(block, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
