import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)
    
    # Build sparse matrix: rows = k-subsets, cols = t-subsets
    # M[i, j] = 1 if k-subset i covers t-subset j
    row_idx = []
    col_idx = []
    for i, t_idxs in enumerate(k_sub_t_idxs):
        for j in t_idxs:
            row_idx.append(i)
            col_idx.append(j)
    
    M = np.zeros((n_k, n_t), dtype=np.int8)
    M[row_idx, col_idx] = 1
    
    # Also build the reverse: for each t-subset, which k-subsets cover it
    # We'll use this for the greedy phase
    
    def time_ok():
        return time.time() < deadline
    
    def greedy_cover(uncovered_mask, M, k_sub_t_idxs, n_t, deadline):
        """
        Greedy covering: repeatedly pick the block that covers the most uncovered t-subsets.
        uncovered_mask: numpy array of shape (n_t,) with 1 for uncovered, 0 for covered.
        Returns list of block indices.
        """
        blocks = []
        uncovered = uncovered_mask.copy()
        n_uncovered = int(uncovered.sum())
        
        while n_uncovered > 0 and time.time() < deadline:
            # Compute gain for each block: number of uncovered t-subsets it covers
            # gain[i] = sum(M[i, j] * uncovered[j] for j in range(n_t))
            gains = M @ uncovered
            # Pick the block with maximum gain
            best_idx = int(np.argmax(gains))
            best_gain = int(gains[best_idx])
            
            if best_gain == 0:
                break  # No block can cover any remaining uncovered t-subset (shouldn't happen)
            
            blocks.append(best_idx)
            
            # Update uncovered
            for j in k_sub_t_idxs[best_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    n_uncovered -= 1
        
        return blocks
    
    def reduce_blocks(blocks, M, k_sub_t_idxs, n_t, deadline):
        """
        Remove redundant blocks. A block is redundant if all its t-subsets are covered by other blocks.
        """
        if not blocks:
            return blocks
        
        # Count how many blocks cover each t-subset
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks:
            for j in k_sub_t_idxs[b_idx]:
                cover_count[j] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                is_essential = False
                for j in k_sub_t_idxs[b_idx]:
                    if cover_count[j] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for j in k_sub_t_idxs[b_idx]:
                        cover_count[j] -= 1
            blocks = new_blocks
        
        return blocks
    
    def local_search(blocks, M, k_sub_t_idxs, n_t, deadline, max_iters=100):
        """
        Try to replace a block with another that covers the same essential t-subsets but allows removing more blocks.
        Simplified: try removing one block at a time, then greedily re-cover if needed.
        """
        best_blocks = blocks[:]
        best_len = len(best_blocks)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try removing each block and see if we can recover
            for i in range(len(best_blocks)):
                if time.time() > deadline:
                    break
                
                candidate = best_blocks[:i] + best_blocks[i+1:]
                
                # Check if candidate still covers everything
                # Compute uncovered t-subsets
                covered = np.zeros(n_t, dtype=np.int8)
                for b_idx in candidate:
                    for j in k_sub_t_idxs[b_idx]:
                        covered[j] = 1
                
                uncovered = 1 - covered
                n_uncovered = int(uncovered.sum())
                
                if n_uncovered == 0:
                    # We can remove this block!
                    best_blocks = candidate
                    best_len = len(best_blocks)
                    break
                elif n_uncovered < 50:  # Small number uncovered, try to greedily fix
                    # Greedily add blocks to cover the remaining
                    extra = greedy_cover(uncovered, M, k_sub_t_idxs, n_t, deadline)
                    if len(extra) == 0:
                        best_blocks = candidate
                        best_len = len(best_blocks)
                        break
                    elif len(extra) < 2:
                        # Adding 1 block to remove 1 is neutral; adding 0 is gain
                        # Only accept if we don't increase block count
                        pass
            else:
                continue
            break  # If we improved, restart the loop
        
        return best_blocks
    
    # Main algorithm
    # Start with an empty covering and greedily build it
    uncovered = np.ones(n_t, dtype=np.int8)
    
    # Greedy phase
    blocks = greedy_cover(uncovered, M, k_sub_t_idxs, n_t, deadline)
    
    # Reduction phase
    blocks = reduce_blocks(blocks, M, k_sub_t_idxs, n_t, deadline)
    
    # Local search phase
    if time.time() < deadline - 20:
        blocks = local_search(blocks, M, k_sub_t_idxs, n_t, deadline)
    
    # Final reduction
    blocks = reduce_blocks(blocks, M, k_sub_t_idxs, n_t, deadline)
    
    # Convert block indices back to blocks (tuples of points)
    result = [all_k_subs[i] for i in blocks]
    
    return result
