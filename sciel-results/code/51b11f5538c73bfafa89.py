import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute coverage for each block
    block_cover_indices = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[sub] for sub in subs], dtype=np.int32)
        block_cover_indices.append(indices)
        
    # Precompute inverse: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, indices in enumerate(block_cover_indices):
        for s_idx in indices:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]

    def greedy_construction(rng, randomized=False, top_k=10, alpha=1.0):
        uncovered = np.ones(n_t_subsets, dtype=bool)
        # Gains: number of uncovered subsets covered by each block
        # Initially, all blocks cover C(k,t) subsets
        gains = np.full(n_k_blocks, len(block_cover_indices[0]), dtype=np.int32)
        
        selected_indices = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if randomized:
                positive_mask = gains > 0
                if not np.any(positive_mask):
                    break
                valid_blocks = np.where(positive_mask)[0]
                if len(valid_blocks) > top_k:
                    valid_gains = gains[valid_blocks]
                    # Soft-max selection using temperature
                    if alpha < 1.0:
                        weights = np.exp(valid_gains * alpha / 10.0)
                        weights /= weights.sum()
                        # Sample from top weighted candidates
                        top_idx = np.argsort(weights)[-top_k:]
                        top_weights = weights[top_idx]
                        top_weights /= top_weights.sum()
                        best_block = valid_blocks[top_idx[rng.choice(len(top_idx), p=top_weights)]]
                    else:
                        top_idx = np.argpartition(valid_gains, -top_k)[-top_k:]
                        top_blocks = valid_blocks[top_idx]
                        weights = gains[top_blocks].astype(float)
                        weights /= weights.sum()
                        best_block = top_blocks[rng.choice(len(top_blocks), p=weights)]
                else:
                    best_block = valid_blocks[rng.integers(0, len(valid_blocks))]
            else:
                best_block = np.argmax(gains)
                if gains[best_block] == 0:
                    break
            
            selected_indices.append(best_block)
            
            # Update gains
            for s_idx in block_cover_indices[best_block]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    # Decrement gain for all blocks covering s_idx
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        return selected_indices

    def local_search(selected_indices, max_rounds=50, swap_passes=1):
        if not selected_indices:
            return selected_indices
            
        selected_set = set(selected_indices)
        
        # Compute cover counts
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in selected_indices:
            counts[block_cover_indices[b_idx]] += 1
            
        improved = True
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline:
            improved = False
            round_num += 1
            
            # Phase A: Redundancy Removal
            new_selected = []
            for b_idx in selected_indices:
                subs = block_cover_indices[b_idx]
                if np.min(counts[subs]) > 1:
                    counts[subs] -= 1
                    improved = True
                else:
                    new_selected.append(b_idx)
            selected_indices = new_selected
            selected_set = set(selected_indices)
            
            # Phase B: Swap Moves
            for _pass in range(swap_passes):
                if time.time() > deadline:
                    break
                for i in range(len(selected_indices)):
                    if time.time() > deadline:
                        break
                        
                    b_idx = selected_indices[i]
                    subs = block_cover_indices[b_idx]
                    unique_mask = counts[subs] == 1
                    unique_subs = subs[unique_mask]
                    
                    if len(unique_subs) == 0:
                        continue
                    
                    first_s = unique_subs[0]
                    candidates = tsub_to_blocks_np[first_s]
                    
                    best_replacement = -1
                    best_score = -1
                    
                    # Try up to 20 candidates
                    for c in candidates[:20]:
                        if c in selected_set:
                            continue
                            
                        # Quick check: does c cover all unique_subs?
                        # Use set intersection for speed
                        c_subs_set = set(block_cover_indices[c].tolist())
                        if not all(s in c_subs_set for s in unique_subs):
                            continue
                        
                        # Score: maximize overlap with existing coverage
                        c_subs = block_cover_indices[c]
                        score = np.sum(counts[c_subs] > 1)
                        
                        if score > best_score:
                            best_score = score
                            best_replacement = c
                            
                    if best_replacement != -1:
                        counts[block_cover_indices[b_idx]] -= 1
                        counts[block_cover_indices[best_replacement]] += 1
                        selected_indices[i] = best_replacement
                        selected_set.discard(b_idx)
                        selected_set.add(best_replacement)
                        improved = True
                        
        return selected_indices

    def verify_solution(selected_indices):
        if not selected_indices:
            return n_t_subsets == 0
        check_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in selected_indices:
            check_counts[block_cover_indices[b_idx]] += 1
        return not np.any(check_counts == 0)

    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts with varying parameters
    num_restarts = 25
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Vary alpha for soft-max, alternate greedy/randomized
        if i % 3 == 0:
            selected_indices = greedy_construction(rng, randomized=False)
        elif i % 3 == 1:
            selected_indices = greedy_construction(rng, randomized=True, top_k=5)
        else:
            selected_indices = greedy_construction(rng, randomized=True, top_k=8, alpha=0.5)
            
        if not selected_indices:
            continue
            
        # Vary local search parameters
        if i % 4 == 0:
            refined_indices = local_search(selected_indices, max_rounds=15, swap_passes=1)
        else:
            refined_indices = local_search(selected_indices, max_rounds=25, swap_passes=2)
        
        if not verify_solution(refined_indices):
            continue
            
        if len(refined_indices) < best_len:
            best_len = len(refined_indices)
            best_blocks = [list(k_blocks[i]) for i in refined_indices]
            
            # Polish if good enough
            if best_len < 145 and time.time() < deadline - 10:
                polished = local_search(refined_indices, max_rounds=40, swap_passes=3)
                if verify_solution(polished) and len(polished) < best_len:
                    best_len = len(polished)
                    best_blocks = [list(k_blocks[i]) for i in polished]

    if best_blocks is None:
        selected_indices = greedy_construction(rng, randomized=False)
        if selected_indices:
            refined_indices = local_search(selected_indices, max_rounds=10, swap_passes=1)
            best_blocks = [list(k_blocks[i]) for i in refined_indices]
        else:
            best_blocks = []
            
    return best_blocks
