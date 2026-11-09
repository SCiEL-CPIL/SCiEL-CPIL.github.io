import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(k_subsets)
    
    block_t_indices = []
    for block in k_subsets:
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(t_sub_to_idx[t_sub])
        block_t_indices.append(indices)
        
    # Precompute reverse map: t_idx -> list of block indices
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, indices in enumerate(block_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    # 2. Greedy Initialization
    def greedy_init():
        cover_count = np.zeros(num_t, dtype=np.int32)
        uncovered_mask = np.ones(num_t, dtype=bool)
        blocks_used = []
        
        while np.any(uncovered_mask):
            if time.time() - start_time > deadline:
                break
            
            uncov_indices = np.where(uncovered_mask)[0]
            if len(uncov_indices) == 0:
                break
            
            # Pick the first uncovered t-subset
            target_t_idx = uncov_indices[0]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in t_to_blocks[target_t_idx]:
                gain = np.sum(uncovered_mask[block_t_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                blocks_used.append(best_block_idx)
                idxs = block_t_indices[best_block_idx]
                cover_count[idxs] += 1
                uncovered_mask[idxs] = (cover_count[idxs] == 0)
                
        return blocks_used

    current_blocks = greedy_init()
    
    # 3. Local Search (ILS)
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            idxs = block_t_indices[b]
            cover_count[idxs] += 1
            
        new_blocks = []
        for b in blocks:
            idxs = block_t_indices[b]
            if np.all(cover_count[idxs] > 1):
                cover_count[idxs] -= 1
            else:
                new_blocks.append(b)
                
        return new_blocks

    def repair(blocks, cover_count):
        uncovered_mask = (cover_count == 0)
        
        while np.any(uncovered_mask):
            if time.time() - start_time > deadline:
                break
            
            uncov_indices = np.where(uncovered_mask)[0]
            if len(uncov_indices) == 0:
                break
                
            target_t_idx = uncov_indices[np.random.choice(len(uncov_indices))]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in t_to_blocks[target_t_idx]:
                gain = np.sum(uncovered_mask[block_t_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                blocks.append(best_block_idx)
                idxs = block_t_indices[best_block_idx]
                cover_count[idxs] += 1
                uncovered_mask[idxs] = (cover_count[idxs] == 0)
                
        return blocks

    best_blocks = reduce_blocks(current_blocks)
    best_len = len(best_blocks)
    
    random.seed(42)
    np.random.seed(42)
    
    while time.time() - start_time < deadline:
        if not best_blocks:
            break
            
        curr_blocks = list(best_blocks)
        
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b in curr_blocks:
            idxs = block_t_indices[b]
            cover_count[idxs] += 1
            
        rem_idx = random.randint(0, len(curr_blocks) - 1)
        rem_block = curr_blocks.pop(rem_idx)
        idxs = block_t_indices[rem_block]
        cover_count[idxs] -= 1
        
        curr_blocks = repair(curr_blocks, cover_count)
        curr_blocks = reduce_blocks(curr_blocks)
        
        if len(curr_blocks) < best_len:
            best_blocks = curr_blocks
            best_len = len(curr_blocks)
            
    return [list(k_subsets[b]) for b in best_blocks]
