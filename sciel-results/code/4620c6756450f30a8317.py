import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for overhead

    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(all_k_blocks)
    
    # Precompute block to t-subset indices
    block_to_t_idxs = []
    # Precompute t-subset index to block indices
    t_idx_to_blocks = [[] for _ in range(num_t_subs)]
    
    for b_idx, block in enumerate(all_k_blocks):
        t_idxs = []
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            t_idxs.append(idx)
            t_idx_to_blocks[idx].append(b_idx)
        block_to_t_idxs.append(t_idxs)

    best_blocks = None
    best_count = float('inf')
    
    # Function to run one iteration (Greedy + Local Search)
    def run_iteration(seed):
        random.seed(seed)
        
        # --- Greedy Phase ---
        uncovered = np.ones(num_t_subs, dtype=bool)
        gains = np.zeros(num_k_blocks, dtype=np.int32)
        
        # Initialize gains
        for s_idx in range(num_t_subs):
            if uncovered[s_idx]:
                for b_idx in t_idx_to_blocks[s_idx]:
                    gains[b_idx] += 1
        
        blocks = []
        
        while True:
            if time.time() > deadline:
                break
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(candidates)
            blocks.append(chosen)
            
            # Update uncovered and gains
            for s_idx in block_to_t_idxs[chosen]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    for b_idx in t_idx_to_blocks[s_idx]:
                        gains[b_idx] -= 1
        
        if not blocks:
            return []

        # --- Local Search Phase ---
        # Initialize cover counts
        cover_count = np.zeros(num_t_subs, dtype=np.int32)
        block_set = set(blocks)
        for b_idx in blocks:
            for s_idx in block_to_t_idxs[b_idx]:
                cover_count[s_idx] += 1
        
        improved = True
        # Limit local search steps to avoid infinite loops if bug, but usually converges fast
        max_ls_steps = 5000 
        ls_steps = 0
        
        while improved and time.time() < deadline and ls_steps < max_ls_steps:
            improved = False
            ls_steps += 1
            
            # Try to remove or swap
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                
                b_idx = blocks[i]
                t_idxs = block_to_t_idxs[b_idx]
                
                # Find unique t-subsets (count == 1)
                unique_idxs = []
                for s_idx in t_idxs:
                    if cover_count[s_idx] == 1:
                        unique_idxs.append(s_idx)
                
                if not unique_idxs:
                    # Remove block
                    blocks.pop(i)
                    block_set.remove(b_idx)
                    for s_idx in t_idxs:
                        cover_count[s_idx] -= 1
                    improved = True
                    break
                else:
                    # Try swaps
                    # Candidates must cover all unique_idxs
                    # Start with candidates for the first unique idx
                    if unique_idxs:
                        candidates = set(t_idx_to_blocks[unique_idxs[0]])
                        for s_idx in unique_idxs[1:]:
                            candidates.intersection_update(t_idx_to_blocks[s_idx])
                        
                        for new_b_idx in candidates:
                            if new_b_idx in block_set:
                                continue
                            
                            # Validity check is implicitly satisfied if new_b_idx covers unique_idxs
                            # Because other s in b_idx have count >= 2, so removing b_idx leaves count >= 1.
                            # Adding new_b_idx only increases counts.
                            
                            # Perform Swap
                            blocks[i] = new_b_idx
                            block_set.remove(b_idx)
                            block_set.add(new_b_idx)
                            
                            # Update cover_count
                            for s_idx in t_idxs:
                                cover_count[s_idx] -= 1
                            for s_idx in block_to_t_idxs[new_b_idx]:
                                cover_count[s_idx] += 1
                            
                            improved = True
                            break
                        if improved:
                            break
                            
        return blocks

    # Run multiple iterations
    iter_count = 0
    while time.time() < deadline:
        seed = iter_count
        blocks = run_iteration(seed)
        
        if blocks:
            count = len(blocks)
            if count < best_count:
                best_count = count
                best_blocks = blocks.copy()
        
        iter_count += 1
        if iter_count % 10 == 0:
            # Optional: print status
            pass

    if best_blocks is None:
        # Fallback to a simple greedy if nothing worked (should not happen)
        return []
        
    # Convert block indices to lists of points
    result = [list(all_k_blocks[b_idx]) for b_idx in best_blocks]
    return result
