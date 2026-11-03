import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_k_blocks = list(itertools.combinations(points, k))
    num_k_blocks = len(all_k_blocks)
    
    # Precompute which t-subsets each k-block covers
    block_subsets = []
    for block in all_k_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_subsets.append(subs)
        
    # Precompute subset_to_blocks
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, subs in enumerate(block_subsets):
        for s in subs:
            subset_to_blocks[s].append(b_idx)

    def greedy_construct(alpha=0.0):
        cover_count = np.zeros(num_t_subsets, dtype=np.int8)
        blocks_in_use = []
        
        while True:
            if time.time() > deadline:
                return None
            
            # Find first uncovered
            uncovered_indices = np.where(cover_count == 0)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Select a few uncovered subsets to define candidate pool
            # To allow randomness, pick 1-3 random uncovered subsets
            num_picks = random.randint(1, 3) if alpha > 0 else 1
            picks = random.sample(uncovered_indices.tolist(), min(num_picks, len(uncovered_indices)))
            
            candidates = set()
            for s in picks:
                candidates.update(subset_to_blocks[s])
            
            if not candidates:
                break
                
            # Score candidates
            best_gain = -1
            best_block = -1
            rcl = []
            
            for b_idx in candidates:
                gain = np.sum(cover_count[block_subsets[b_idx]] == 0)
                if gain > 0:
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                        rcl = [(gain, b_idx)]
                    elif gain == best_gain:
                        rcl.append((gain, b_idx))
            
            if best_block == -1:
                # Should not happen
                break
                
            if alpha > 0 and len(rcl) > 1:
                selected = random.choice(rcl)[1]
            else:
                selected = best_block
                
            blocks_in_use.append(selected)
            cover_count[block_subsets[selected]] += 1
            
        return blocks_in_use

    def local_search(blocks_in_use):
        if blocks_in_use is None:
            return None
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int8)
        for b in blocks_in_use:
            cover_count[block_subsets[b]] += 1
            
        current_blocks = list(blocks_in_use)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_blocks)
            
            for i in range(len(current_blocks) - 1, -1, -1):
                b_idx = current_blocks[i]
                
                # Check redundancy
                if np.all(cover_count[block_subsets[b_idx]] > 1):
                    cover_count[block_subsets[b_idx]] -= 1
                    current_blocks.pop(i)
                    improved = True
                    continue
                    
                # Find unique subsets
                unique_subsets = [s for s in block_subsets[b_idx] if cover_count[s] == 1]
                
                if not unique_subsets:
                    continue

                # Try to find replacement
                # 1. Check existing blocks
                found_existing = False
                for j in range(len(current_blocks)):
                    if j == i:
                        continue
                    other_b = current_blocks[j]
                    if all(s in block_subsets[other_b] for s in unique_subsets):
                        cover_count[block_subsets[b_idx]] -= 1
                        current_blocks.pop(i)
                        improved = True
                        found_existing = True
                        break
                
                if found_existing:
                    continue

                # 2. Check new blocks (only if unique_subsets is small)
                if len(unique_subsets) <= 2:
                    # Intersect candidate lists
                    cand_set = set(subset_to_blocks[unique_subsets[0]])
                    for s in unique_subsets[1:]:
                        cand_set.intersection_update(subset_to_blocks[s])
                    
                    for cand_b in cand_set:
                        if cand_b == b_idx:
                            continue
                        # Verify (should be covered by construction)
                        cover_count[block_subsets[b_idx]] -= 1
                        current_blocks.pop(i)
                        current_blocks.append(cand_b)
                        cover_count[block_subsets[cand_b]] += 1
                        improved = True
                        break
                            
        return current_blocks

    best_solution = None
    best_len = float('inf')
    
    while time.time() < deadline:
        alpha = random.uniform(0, 0.5)
        blocks = greedy_construct(alpha)
        
        if blocks is None:
            break
            
        blocks = local_search(blocks)
        
        if blocks is not None:
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_solution = blocks
                if best_len <= 330:
                    break

    if best_solution is None:
        best_solution = greedy_construct(0.0)
        best_solution = local_search(best_solution)
        
    result = [list(all_k_blocks[b]) for b in best_solution]
    return result
