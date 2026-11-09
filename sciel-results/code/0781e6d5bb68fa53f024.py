import numpy as np
from itertools import combinations
import heapq

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subs = list(combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute all k-subsets
    k_subs = list(combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build forward mapping: block_idx -> list of t_sub_idx
    block_t_indices = []
    for ks in k_subs:
        indices = []
        for ts in combinations(ks, t):
            indices.append(t_sub_idx[ts])
        block_t_indices.append(indices)
        
    # Build reverse mapping: t_sub_idx -> list of block_idx
    # Initialize list of empty lists
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, t_indices in enumerate(block_t_indices):
        for t_idx in t_indices:
            t_to_blocks[t_idx].append(b_idx)
            
    # Initial gains: each block covers C(k, t) t-subsets
    gains = np.full(n_k, len(block_t_indices[0]), dtype=np.int32)
    
    # Heap of (-gain, block_idx)
    heap = [(-gains[i], i) for i in range(n_k)]
    heapq.heapify(heap)
    
    # Track used blocks to avoid re-adding
    used = np.zeros(n_k, dtype=bool)
    
    # Uncovered mask
    uncovered = np.ones(n_t, dtype=bool)
    n_uncovered = n_t
    
    blocks = []
    
    while n_uncovered > 0:
        # Pop best block
        while heap:
            neg_gain, b_idx = heapq.heappop(heap)
            gain = -neg_gain
            
            # Lazy deletion: check if this entry is stale
            if used[b_idx]:
                continue
            if gain != gains[b_idx]:
                continue
            if gain == 0:
                # If max gain is 0, no more progress possible (should not happen if valid)
                break
            
            # Select this block
            blocks.append(k_subs[b_idx])
            used[b_idx] = True
            
            # Update uncovered and gains
            for t_idx in block_t_indices[b_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    n_uncovered -= 1
                    # Update gains for all blocks covering this t_idx
                    for other_b_idx in t_to_blocks[t_idx]:
                        if not used[other_b_idx]:
                            gains[other_b_idx] -= 1
                            heapq.heappush(heap, (-gains[other_b_idx], other_b_idx))
        
        if not heap or (heap and -heap[0][0] == 0):
            # If heap is empty or max gain is 0, break
            # Note: if n_uncovered > 0 and max gain is 0, it's an error in logic or data
            break

    return [list(b) for b in blocks]
