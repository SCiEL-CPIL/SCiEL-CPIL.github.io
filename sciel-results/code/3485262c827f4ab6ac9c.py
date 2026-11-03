import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute all blocks and their triple coverage
    all_blocks_list = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks_list)
    
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {tup: i for i, tup in enumerate(all_triples)}
    
    # Precompute block_triples_arr
    block_triples_list = []
    for b in all_blocks_list:
        idxs = [triple_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_triples_list.append(np.array(idxs, dtype=np.int32))
    block_triples_arr = np.array(block_triples_list, dtype=np.int32)
    
    # Precompute incidence matrix
    rows = []
    cols = []
    for i, bt in enumerate(block_triples_arr):
        for j in bt:
            rows.append(i)
            cols.append(j)
    inc_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_blocks, n_triples))
    
    best_blocks = None
    best_score = float('inf')
    
    max_restarts = 1000
    
    for run in range(max_restarts):
        if time.time() > deadline:
            break
        
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            gains = (inc_matrix @ uncovered).A1
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_block_idx = np.random.choice(candidates)
            
            blocks.append(best_block_idx)
            uncovered[block_triples_arr[best_block_idx]] = False
        
        if not uncovered.any():
            # Post-processing
            coverage_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in blocks:
                coverage_count[block_triples_arr[b_idx]] += 1
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_blocks = []
                for b_idx in blocks:
                    if np.min(coverage_count[block_triples_arr[b_idx]]) > 1:
                        coverage_count[block_triples_arr[b_idx]] -= 1
                        changed = True
                    else:
                        new_blocks.append(b_idx)
                blocks = new_blocks
            
            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = [all_blocks_list[i] for i in blocks]
                
    if best_blocks is None:
        # Fallback
        uncovered = np.ones(n_triples, dtype=bool)
        blocks = []
        while uncovered.any():
            gains = (inc_matrix @ uncovered).A1
            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            best_block_idx = candidates[0]
            blocks.append(best_block_idx)
            uncovered[block_triples_arr[best_block_idx]] = False
        best_blocks = [all_blocks_list[i] for i in blocks]

    return best_blocks
