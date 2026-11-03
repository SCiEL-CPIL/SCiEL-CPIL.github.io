import itertools
import numpy as np
import math

def construct(v, k, t):
    points = range(v)
    blocks_k = list(itertools.combinations(points, k))
    subsets_t = list(itertools.combinations(points, t))
    
    n_blocks = len(blocks_k)
    n_subsets = len(subsets_t)
    
    # Map subset to index
    sub_to_idx = {s: i for i, s in enumerate(subsets_t)}
    
    # Precompute which blocks cover which subsets
    # covering_blocks[i] is a list of block indices that cover subsets_t[i]
    covering_blocks = [[] for _ in range(n_subsets)]
    
    for b_idx, b in enumerate(blocks_k):
        for sub in itertools.combinations(b, t):
            s_idx = sub_to_idx[sub]
            covering_blocks[s_idx].append(b_idx)
            
    # Initialize gains
    # Each block covers C(k, t) subsets
    initial_gain = math.comb(k, t)
    gains = np.full(n_blocks, initial_gain, dtype=np.int32)
    
    # Set of uncovered subset indices
    uncovered = set(range(n_subsets))
    
    selected_blocks = []
    
    while uncovered:
        # Find the block with the maximum gain
        best_idx = np.argmax(gains)
        
        # Add this block to the solution
        selected_blocks.append(list(blocks_k[best_idx]))
        
        # Update gains for subsets covered by this block
        best_block = blocks_k[best_idx]
        for sub in itertools.combinations(best_block, t):
            s_idx = sub_to_idx[sub]
            if s_idx in uncovered:
                # This subset is now covered
                uncovered.remove(s_idx)
                
                # Decrement gain for all blocks that cover this subset
                for b_idx in covering_blocks[s_idx]:
                    gains[b_idx] -= 1
                    
    return selected_blocks
