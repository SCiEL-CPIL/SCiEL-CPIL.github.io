import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for safety
    
    # 1. Generate all k-blocks and t-subsets
    points = range(v)
    k_blocks = list(itertools.combinations(points, k))
    t_subs = list(itertools.combinations(points, t))
    
    num_blocks = len(k_blocks)
    num_subs = len(t_subs)
    
    # Map subset to index
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # For each block, list the indices of the t-subsets it covers
    block_subs = []
    for block in k_blocks:
        subs_in_block = [sub_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_subs.append(subs_in_block)
        
    # For each subset, list the indices of the blocks that cover it
    sub_blocks = [[] for _ in range(num_subs)]
    for i, subs in enumerate(block_subs):
        for j in subs:
            sub_blocks[j].append(i)

    # 2. Greedy Construction
    covered_count = np.zeros(num_subs, dtype=np.int32)
    remaining_uncovered = num_subs
    solution = []
    used_blocks = np.zeros(num_blocks, dtype=bool)
    
    # Precompute block_subs as numpy arrays for faster iteration? 
    # Actually, list of lists is fine if we iterate carefully.
    # Let's try to speed up the greedy step using numpy where possible, 
    # but the "find max gain" step is hard to vectorize easily without huge memory.
    
    # Optimization: 
    # Instead of checking all blocks every time, we can maintain a list of active blocks.
    # But for v=19, num_blocks is ~50k. 50k * 35 = 1.75M ops per step.
    # 150 steps * 1.75M = 262M ops. This is the bottleneck.
    
    # Let's try a different greedy strategy or speed up the inner loop.
    # We can use a heap? No, gains change.
    
    # Alternative: Randomized greedy with multiple restarts? 
    # Or just optimize the current one.
    
    # Let's try to use a list of indices for active blocks.
    active_blocks = list(range(num_blocks))
    
    while remaining_uncovered > 0:
        if time.time() > deadline - 20:
            break
            
        best_block = -1
        best_gain = 0
        
        # Iterate through active blocks
        for i in active_blocks:
            if used_blocks[i]:
                continue
            
            # Calculate gain
            # Sum of 1 for s in block_subs[i] if covered_count[s] == 0
            # This is len(block_subs[i]) - sum(1 for s in block_subs[i] if covered_count[s] > 0)
            
            gain = 0
            for s_idx in block_subs[i]:
                if covered_count[s_idx] == 0:
                    gain += 1
                    # Early break if gain cannot exceed best_gain?
                    # Max possible gain is len(block_subs[i])
                    # If gain + (len(block_subs[i]) - (current_index+1)) <= best_gain, break?
                    # But we don't track index easily in this loop.
                    # Just compute full gain.
            
            if gain > best_gain:
                best_gain = gain
                best_block = i
        
        if best_block == -1 or best_gain == 0:
            break
            
        # Add best_block
        solution.append(best_block)
        used_blocks[best_block] = True
        
        # Update covered_count
        for s_idx in block_subs[best_block]:
            if covered_count[s_idx] == 0:
                remaining_uncovered -= 1
            covered_count[s_idx] += 1

    # 3. Local Search
    # We want to minimize len(solution).
    # Strategy:
    # 1. Remove redundant blocks.
    # 2. Swap blocks to enable further removals.
    
    improved = True
    max_iters = 100
    iter_count = 0
    
    while improved and iter_count < max_iters and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        
        # Pass 1: Remove redundant blocks
        for i in list(solution):
            # Check if any subset in block i has count == 1
            # If min count > 1, remove it
            # Optimization: Use numpy min?
            # min_count = np.min(covered_count[block_subs[i]])
            # But block_subs[i] is a list.
            
            min_count = min(covered_count[s_idx] for s_idx in block_subs[i])
            if min_count > 1:
                # Remove i
                for s_idx in block_subs[i]:
                    covered_count[s_idx] -= 1
                solution.remove(i)
                improved = True
                break # Restart scan to find new redundancies
        
        if improved:
            continue
        
        # Pass 2: Swap blocks
        sol_set = set(solution)
        
        for i in solution:
            if time.time() > deadline - 5:
                break
                
            D = []
            for s_idx in block_subs[i]:
                if covered_count[s_idx] == 1:
                    D.append(s_idx)
            
            if not D:
                continue
                
            # Find candidate blocks j that cover all of D
            D_sorted = sorted(D, key=lambda x: len(sub_blocks[x]))
            
            if not D_sorted:
                continue
                
            candidates = set(sub_blocks[D_sorted[0]])
            
            for s_idx in D_sorted[1:]:
                candidates.intersection_update(sub_blocks[s_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            valid_candidates = [j for j in candidates if j not in sol_set]
            
            if not valid_candidates:
                continue
            
            best_j = -1
            best_score = -1
            
            for j in valid_candidates:
                score = 0
                for s_idx in block_subs[j]:
                    if covered_count[s_idx] == 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_j = j
                    
            if best_j != -1:
                # Perform swap
                for s_idx in block_subs[i]:
                    covered_count[s_idx] -= 1
                sol_set.remove(i)
                solution.remove(i)
                
                for s_idx in block_subs[best_j]:
                    covered_count[s_idx] += 1
                sol_set.add(best_j)
                solution.append(best_j)
                
                improved = True
                break # Restart scan

    # Return the list of blocks
    return [list(k_blocks[i]) for i in solution]
