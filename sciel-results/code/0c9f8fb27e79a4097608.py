import itertools
import time
import random
import sys
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    """
    Solve the covering design problem C(v, k, t) using a deterministic greedy 
    set cover approach followed by local search for block removal and swapping.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin from 120s limit
    
    # Precompute all t-subsets (the universe to be covered)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each potential block (k-subset), which t-subsets it covers
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_covered_t = [None] * num_blocks
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for i, b in enumerate(all_blocks):
        covered_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx[ts]
            covered_indices.append(idx)
            t_to_blocks[idx].append(i)
        block_covered_t[i] = covered_indices
    
    # Convert to numpy arrays for faster operations
    # For each block, store its covered t-subset indices as a numpy array
    block_covered_t_np = [np.array(block_covered_t[i], dtype=np.int32) for i in range(num_blocks)]
    
    # Greedy Cover Function - deterministic: pick block covering most uncovered
    def run_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            # Find block covering most uncovered t-subsets
            best_block = -1
            best_gain = 0
            
            # Sample candidates for efficiency when many blocks exist
            if num_blocks > 5000:
                # Focus on blocks that cover at least one uncovered t-subset
                # Use random sampling for speed
                sample_size = min(2000, num_blocks)
                candidates = random.sample(range(num_blocks), sample_size)
            else:
                candidates = range(num_blocks)
            
            for b_idx in candidates:
                if time.time() > deadline:
                    break
                covered = block_covered_t_np[b_idx]
                gain = np.count_nonzero(uncovered[covered])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block == -1:
                break
            
            selected.append(best_block)
            uncovered[block_covered_t_np[best_block]] = False
        
        return selected
    
    # Local Search: Remove Redundant Blocks
    def try_remove_blocks(blocks):
        if not blocks:
            return blocks
        
        # Build coverage counts using numpy
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in blocks:
            cover_count[block_covered_t_np[b_idx]] += 1
        
        # Greedy removal:
        changed = True
        while changed and time.time() < deadline - 10.0:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 10.0:
                    return blocks
                
                b_idx = blocks[i]
                covered = block_covered_t_np[b_idx]
                # Block is redundant if all its t-subsets have count > 1
                if np.all(cover_count[covered] > 1):
                    cover_count[covered] -= 1
                    blocks.pop(i)
                    changed = True
                    break  # Restart loop since indices changed
        
        return blocks
    
    # Local Search: Swap-based Improvement
    def local_search_swap(blocks, max_iters=500):
        """
        Try to improve the solution by swapping one block with another
        that covers all its essential t-subsets while minimizing loss.
        """
        if not blocks:
            return blocks
        
        current_blocks = list(blocks)
        
        # Build coverage counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_covered_t_np[b_idx]] += 1
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iters and time.time() < deadline - 15.0:
            improved = False
            iterations += 1
            
            # First, try to remove redundant blocks
            removed_any = False
            for i in range(len(current_blocks)):
                if time.time() > deadline - 15.0:
                    break
                b_idx = current_blocks[i]
                covered = block_covered_t_np[b_idx]
                if np.all(cover_count[covered] > 1):
                    cover_count[covered] -= 1
                    current_blocks.pop(i)
                    improved = True
                    removed_any = True
                    break
            
            if removed_any:
                continue
            
            # Try swapping: pick a block, try to replace it with another
            # that covers all its essential t-subsets
            for i in range(len(current_blocks)):
                if time.time() > deadline - 15.0:
                    break
                
                b_idx = current_blocks[i]
                covered_b = block_covered_t_np[b_idx]
                
                # Find t-subsets only covered by this block (essential)
                essential_mask = cover_count[covered_b] == 1
                essential_t = covered_b[essential_mask]
                
                # If no essential t-subsets, this block is redundant
                if len(essential_t) == 0:
                    cover_count[covered_b] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block
                best_swap_block = -1
                best_score = 0  # Higher is better
                
                # Sample candidate blocks
                sample_size = min(3000, num_blocks)
                candidates = random.sample(range(num_blocks), sample_size)
                
                for cand_idx in candidates:
                    if cand_idx == b_idx:
                        continue
                    
                    covered_c = block_covered_t_np[cand_idx]
                    
                    # Check if candidate covers all essential t-subsets
                    # Use set intersection for speed
                    if not set(essential_t).issubset(set(covered_c)):
                        continue
                    
                    # Score: number of new t-subsets covered (not in b_idx)
                    # with low coverage count, minus penalty for losing unique coverage
                    new_t = np.setdiff1d(covered_c, covered_b)
                    lost_t = np.setdiff1d(covered_b, covered_c)
                    
                    # Gain from new t-subsets with low coverage
                    if len(new_t) > 0:
                        gain = np.sum(cover_count[new_t] <= 2)
                    else:
                        gain = 0
                    
                    # Penalty for losing t-subsets that had low coverage
                    if len(lost_t) > 0:
                        penalty = np.sum(cover_count[lost_t] <= 2)
                    else:
                        penalty = 0
                    
                    # Also count how many essential t-subsets are still covered
                    # (they are, by construction)
                    
                    score = gain - 2 * penalty  # Heavy penalty for losing coverage
                    
                    if score > best_score:
                        best_score = score
                        best_swap_block = cand_idx
                
                if best_swap_block != -1 and best_score > 0:
                    # Perform the swap
                    cover_count[covered_b] -= 1
                    cover_count[block_covered_t_np[best_swap_block]] += 1
                    current_blocks[i] = best_swap_block
                    improved = True
                    break
        
        return current_blocks
    
    # Main Search Loop
    best_blocks = None
    best_len = float('inf')
    
    # Try multiple seeds with deterministic greedy
    seeds = [42, 123, 999, 0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 31, 37, 41, 43, 47]
    
    for seed in seeds:
        if time.time() > deadline - 25.0:
            break
        
        greedy_res = run_greedy(seed)
        if not greedy_res:
            continue
        
        # Post-process: first remove redundant blocks, then try local search
        processed_res = try_remove_blocks(greedy_res)
        ls_res = local_search_swap(processed_res)
        
        if len(ls_res) < best_len:
            best_len = len(ls_res)
            best_blocks = ls_res
        
        if best_len <= 69:
            break
    
    # If we have time, try more random seeds
    if time.time() < deadline - 20.0:
        while time.time() < deadline - 10.0:
            seed = random.randint(0, 1000000)
            greedy_res = run_greedy(seed)
            if not greedy_res:
                continue
            
            processed_res = try_remove_blocks(greedy_res)
            ls_res = local_search_swap(processed_res)
            
            if len(ls_res) < best_len:
                best_len = len(ls_res)
                best_blocks = ls_res
                
            if best_len <= 69:
                break
    
    if best_blocks is None:
        return []
    
    # Convert block indices back to the actual blocks (lists of points)
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    # Ensure distinctness and sorting
    for b in final_blocks:
        b.sort()
    
    return final_blocks
