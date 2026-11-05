import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s

    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)

    # Pre-compute all k-subsets (blocks) and their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_t_subs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_t_subs.append(subs)
    
    # Pre-compute subset_to_blocks for fast updates
    subset_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_t_subs):
        for s in subs:
            subset_to_blocks[s].append(b_idx)
    
    # Convert subset_to_blocks to sorted lists for faster intersection
    for s in range(n_t_subs):
        subset_to_blocks[s].sort()
    
    # Pre-compute set of t-subsets for each block for O(1) membership
    block_t_sub_sets = [set(subs) for subs in block_t_subs]
            
    # --- PHASE 1: Greedy Construction ---
    # We use a "most uncovered" greedy strategy.
    # uncovered_count_per_block[i] = number of subsets in block i that are currently uncovered.
    uncovered = [1] * n_t_subs
    uncovered_count_per_block = [len(subs) for subs in block_t_subs]
    total_uncovered = n_t_subs
    selected_blocks = []
    
    # Optimization: Instead of scanning all blocks every step, we can just scan.
    # 50k blocks * 150 steps = 7.5M ops. This is fast in Python if optimized.
    # But we can do better by maintaining a list of active blocks? No, simple scan is robust.
    
    while total_uncovered > 0:
        if time.time() - start_time > time_limit * 0.4:
            break
            
        # Find block with max uncovered count
        max_count = 0
        best_b_idx = -1
        for i in range(n_blocks):
            c = uncovered_count_per_block[i]
            if c > max_count:
                max_count = c
                best_b_idx = i
                if max_count == len(block_t_subs[i]):
                    break # Perfect block found, can't do better
            
        if best_b_idx == -1 or max_count == 0:
            break
            
        selected_blocks.append(best_b_idx)
        
        # Update uncovered status and counts
        for s in block_t_subs[best_b_idx]:
            if uncovered[s]:
                uncovered[s] = 0
                total_uncovered -= 1
                # Update counts for all blocks containing s
                for b_idx in subset_to_blocks[s]:
                    uncovered_count_per_block[b_idx] -= 1

    # --- PHASE 2: Local Search / Reduction ---
    # Initialize cover_count based on selected_blocks
    cover_count = [0] * n_t_subs
    for b_idx in selected_blocks:
        for s in block_t_subs[b_idx]:
            cover_count[s] += 1
    
    # Precompute which blocks contain each t-subset as a set for fast lookup
    # Actually, subset_to_blocks already gives us this. Let's keep it as lists.
    
    # For fast "is block in selected" checks
    selected_set = set(selected_blocks)
            
    # 2a. Remove Redundant Blocks
    # A block is redundant if all its subsets are covered by other blocks (cover_count > 1 for all its subsets, considering the block's own contribution)
    # Actually, simpler: if for all s in block, cover_count[s] > 1, then removing it doesn't uncover anything.
    # Wait, cover_count includes the block itself. So if cover_count[s] >= 2 for all s in block, it's redundant.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.7:
        improved = False
        for i in range(len(selected_blocks)):
            b_idx = selected_blocks[i]
            redundant = True
            for s in block_t_subs[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for s in block_t_subs[b_idx]:
                    cover_count[s] -= 1
                selected_blocks.pop(i)
                improved = True
                break # Restart loop as list changed

    # 2b. 1-for-1 Swaps (Replace 1 block with another that covers its critical subsets)
    # For a block b, if there exists b_new not in selected such that b_new covers all
    # critical subsets of b (subsets s in b with cover_count[s]==1), we can replace b with b_new.
    # This doesn't change the count, but may enable further reductions.
    
    # 2c. 2-for-1 Swaps (Replace 2 blocks with 1)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.95:
        improved = False
        n_sel = len(selected_blocks)
        selected_set = set(selected_blocks)
        
        # --- 1-for-1 swap pass ---
        for i in range(n_sel):
            if time.time() - start_time > time_limit * 0.95:
                break
            b1 = selected_blocks[i]
            crit1 = [s for s in block_t_subs[b1] if cover_count[s] == 1]
            
            if len(crit1) == 0:
                continue  # Should be caught by 2a, but safety
            
            # Find candidates covering all of crit1
            # Start with the smallest subset_to_blocks list
            crit1_sorted = sorted(crit1, key=lambda s: len(subset_to_blocks[s]))
            candidates = list(subset_to_blocks[crit1_sorted[0]])
            for s in crit1_sorted[1:]:
                if not candidates:
                    break
                # Intersect with subset_to_blocks[s]
                # Since both are sorted, we can do a fast merge-intersection
                new_candidates = []
                j = 0
                target = subset_to_blocks[s]
                for c in candidates:
                    while j < len(target) and target[j] < c:
                        j += 1
                    if j < len(target) and target[j] == c:
                        new_candidates.append(c)
                candidates = new_candidates
            
            best_new = -1
            for b_new in candidates:
                if b_new in selected_set:
                    continue
                best_new = b_new
                break
            
            if best_new != -1:
                # Perform the 1-for-1 swap
                for s in block_t_subs[b1]:
                    cover_count[s] -= 1
                for s in block_t_subs[best_new]:
                    cover_count[s] += 1
                
                selected_blocks[i] = best_new
                selected_set.discard(b1)
                selected_set.add(best_new)
                
                improved = True
                break
        
        if improved:
            continue
        
        # --- 2-for-1 swap pass ---
        n_sel = len(selected_blocks)
        selected_set = set(selected_blocks)
        
        for i in range(n_sel):
            if time.time() - start_time > time_limit * 0.95:
                break
            b1 = selected_blocks[i]
            crit1 = [s for s in block_t_subs[b1] if cover_count[s] == 1]
            
            if len(crit1) == 0:
                continue
            
            for j in range(i + 1, n_sel):
                b2 = selected_blocks[j]
                crit2 = [s for s in block_t_subs[b2] if cover_count[s] == 1]
                
                if len(crit2) == 0:
                    continue
                
                # Combine critical subsets: L = crit1 ∪ crit2
                # Use a set to avoid duplicates
                L = set(crit1)
                L.update(crit2)
                
                if not L:
                    continue
                
                # Find candidates covering all of L
                L_list = list(L)
                L_sorted = sorted(L_list, key=lambda s: len(subset_to_blocks[s]))
                candidates = list(subset_to_blocks[L_sorted[0]])
                for s in L_sorted[1:]:
                    if not candidates:
                        break
                    new_candidates = []
                    jj = 0
                    target = subset_to_blocks[s]
                    for c in candidates:
                        while jj < len(target) and target[jj] < c:
                            jj += 1
                        if jj < len(target) and target[jj] == c:
                            new_candidates.append(c)
                    candidates = new_candidates
                
                best_new = -1
                for b_new in candidates:
                    if b_new in selected_set:
                        continue
                    best_new = b_new
                    break
                
                if best_new != -1:
                    # Perform the 2-for-1 swap
                    for s in block_t_subs[b1]:
                        cover_count[s] -= 1
                    for s in block_t_subs[b2]:
                        cover_count[s] -= 1
                    for s in block_t_subs[best_new]:
                        cover_count[s] += 1
                    
                    # Update selected_blocks
                    selected_blocks.pop(j)
                    selected_blocks.pop(i)
                    selected_blocks.append(best_new)
                    selected_set.discard(b1)
                    selected_set.discard(b2)
                    selected_set.add(best_new)
                    
                    improved = True
                    break
                    
            if improved:
                break
        
        # After swaps, do another removal pass
        if improved:
            improved = True
            while improved and (time.time() - start_time) < time_limit * 0.95:
                improved = False
                for i in range(len(selected_blocks)):
                    b_idx = selected_blocks[i]
                    redundant = True
                    for s in block_t_subs[b_idx]:
                        if cover_count[s] == 1:
                            redundant = False
                            break
                    if redundant:
                        for s in block_t_subs[b_idx]:
                            cover_count[s] -= 1
                        selected_blocks.pop(i)
                        improved = True
                        break
                        
    final_blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    # Final validation (optional, but good for debugging)
    # If time permits, check validity
    # For now, trust the logic.
    
    return final_blocks
