import itertools
import time
import random
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k = len(all_k_blocks)
    
    # Build sparse matrix M: M[b, s] = 1 if block b covers t-subset s
    rows = []
    cols = []
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[t_sub]
            rows.append(b_idx)
            cols.append(s_idx)
            
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(num_k, num_t))
    
    # Also keep dense lists of indices for fast access if needed, but sparse is key for gain
    # M is CSR, so M[b] gives a sparse row.
    
    best_blocks = None
    best_score = float('inf')
    
    def reduce_blocks(blocks_idx, coverage):
        """Remove redundant blocks. Returns new list of block indices."""
        changed = True
        while changed:
            changed = False
            new_blocks = []
            # We need to check each block in the current list
            # To make this faster, we can process in random order or just iterate
            for b_idx in blocks_idx:
                # Get indices of t-subsets covered by this block
                # M[b_idx] is a sparse row. We need its non-zero indices.
                # Since M is CSR, M[b_idx].indices gives the column indices.
                idxs = M[b_idx].indices
                # Check if coverage is > 1 for all these idxs
                if np.all(coverage[idxs] > 1):
                    # Safe to remove
                    coverage[idxs] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks_idx = new_blocks
        return blocks_idx

    def greedy_construct(seed):
        random.seed(seed)
        coverage = np.zeros(num_t, dtype=np.int32)
        uncovered_count = num_t
        blocks_idx = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                return None, coverage, blocks_idx
            
            # Compute gain for all blocks
            # gain[b] = number of uncovered t-subsets covered by block b
            # uncovered_mask = (coverage == 0)
            # We can compute this efficiently with sparse matrix multiplication?
            # M is (num_k, num_t). coverage is (num_t,).
            # We want sum(M[b, :] * (coverage == 0)) for each b.
            
            # Let's use dense for the mask if num_t is small enough?
            # num_t = 11628. num_k = 75582.
            # M is sparse.
            # Let's try:
            # mask = (coverage == 0).astype(np.int8)
            # gains = M.dot(mask)
            
            mask = (coverage == 0)
            # M is int8. mask is bool. 
            # M.dot(mask) will sum 1s where mask is True.
            gains = M.dot(mask.astype(np.int8))
            
            # Find block with max gain
            # If max gain is 0, we are stuck (should not happen if uncovered_count > 0)
            max_gain = np.max(gains)
            if max_gain == 0:
                return None, coverage, blocks_idx
                
            # Pick one of the blocks with max gain (random tie-breaking)
            candidates = np.where(gains == max_gain)[0]
            b_idx = random.choice(candidates)
            
            blocks_idx.append(b_idx)
            
            # Update coverage
            # coverage += M[b_idx]
            # M[b_idx] is sparse. We need to convert to dense or use indices.
            idxs = M[b_idx].indices
            coverage[idxs] += 1
            
            # Update uncovered_count
            # newly covered = idxs where coverage was 0 before update
            # We can just recount or track.
            # Recounting is O(num_t). num_t is 11628. This is fast.
            uncovered_count = np.sum(coverage == 0)
            
        return blocks_idx, coverage, None

    # Run multiple restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        res = greedy_construct(i)
        if res[0] is None:
            continue
            
        blocks_idx, coverage, _ = res
        
        # Reduce
        blocks_idx = reduce_blocks(blocks_idx, coverage)
        
        current_score = len(blocks_idx)
        if current_score < best_score:
            best_score = current_score
            best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
            
            # Local Search
            ls_start = time.time()
            # Try to improve by swapping or adding/removing
            # Simple hill climbing: try removing a block, then greedily cover gaps with 1 block?
            # Or try adding a block and reducing.
            
            # Let's try a simple local search:
            # 1. Try to remove each block, if it breaks coverage, try to fix with one new block.
            # This is expensive.
            
            # Alternative: Randomized Local Search
            # Pick a block to remove, pick a block to add, reduce.
            
            for _ in range(50): # Limit local search steps
                if time.time() > deadline - 2.0:
                    break
                
                # Try adding a random block and reducing
                b_add = random.randint(0, num_k - 1)
                if b_add in blocks_idx:
                    continue
                
                old_blocks = list(blocks_idx)
                old_coverage = coverage.copy()
                
                blocks_idx.append(b_add)
                idxs = M[b_add].indices
                coverage[idxs] += 1
                
                new_blocks = reduce_blocks(blocks_idx, coverage)
                
                if len(new_blocks) < len(old_blocks):
                    blocks_idx = new_blocks
                    best_score = len(blocks_idx)
                    best_blocks = [list(all_k_blocks[b]) for b in blocks_idx]
                else:
                    blocks_idx = old_blocks
                    coverage = old_coverage
                    
    return best_blocks
