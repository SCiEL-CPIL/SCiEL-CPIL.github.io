import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
    
    # Precompute block sizes (number of t-subsets each block covers) for efficiency scoring
    block_sizes = [len(subs) for subs in block_to_tsubs]
    
    # Build bitsets for fast gain calculation
    # Each block covers a set of t-subsets. We represent this as a Python int bitmask.
    # However, Python ints can be arbitrarily large, and bitwise ops are fast.
    # n_t_subs for C(19,4) is 3876. This fits in a 64-bit integer? No, 2^3876 is huge.
    # But Python handles big ints. Let's see if it's fast enough.
    # Alternatively, we can use numpy arrays for the "uncovered" state and vectorize the gain calculation.
    # Let's stick to the list-based approach but optimize the inner loop by pre-computing
    # the list of t-subsets for each block as a numpy array for faster membership checks?
    # Actually, the bottleneck in the current greedy phase is:
    # for i in range(n_blocks):
    #     for s in block_to_tsubs[i]:
    #         if uncovered[s]: ...
    # This is ~50k blocks * 15 t-subsets = 750k checks per greedy step.
    # With ~140 blocks, that's ~100M checks. Python might be too slow for 115s if not optimized.
    # Let's try a different approach for the greedy phase:
    # Maintain a count of how many times each t-subset is covered.
    # When we pick a block, we decrement the count for all t-subsets it covers.
    # For the next pick, we want to maximize the number of "newly" covered t-subsets (count==0).
    # This is equivalent to maximizing the intersection of the block's t-subsets with the currently uncovered set.
    
    # To speed this up, we can pre-compute for each block a list of its t-subsets.
    # And maintain a boolean array `uncovered` of length n_t_subs.
    # The gain of block i is sum(1 for s in block_to_tsubs[i] if uncovered[s]).
    # We can compute this efficiently if we store block_to_tsubs as numpy arrays?
    # Or just use Python lists and hope it's fast enough with the reduced number of iterations?
    # Let's try to optimize the greedy step by only considering blocks that cover at least one uncovered subset.
    
    # Even better: Use a "dual" weight approach where we maintain weights for uncovered subsets.
    # But let's first try to just make the current approach faster or use a different heuristic.
    
    # Alternative: Randomized Greedy with "Conflicts"
    # Let's use a simpler, faster greedy:
    # 1. Start with all subsets uncovered.
    # 2. Repeatedly pick the block that covers the most uncovered subsets.
    #    To break ties, pick the one that covers the "rarest" uncovered subsets (those with fewest covering blocks).
    
    # Let's pre-compute the "rarity" of each t-subset (how many blocks cover it).
    tsub_rarity = [len(blocks) for blocks in tsub_to_blocks]
    
    # For speed, let's use numpy for the uncovered array and try to vectorize the gain calculation.
    # But iterating over 50k blocks in Python is the bottleneck.
    # Let's reduce the search space in the greedy phase by only considering "good" blocks.
    # Or, we can use a randomized approach: pick a random subset of candidate blocks.
    
    # Let's try a hybrid:
    # 1. Compute the gain for all blocks.
    # 2. Select the top M blocks by gain.
    # 3. Pick one randomly from the top M (or the best one).
    
    # To compute gain for all blocks, we can do:
    # gains = np.zeros(n_blocks)
    # for b_idx in range(n_blocks):
    #     gains[b_idx] = np.sum(uncovered[np.array(block_to_tsubs[b_idx])])
    # This is still O(n_blocks * t) which is slow in Python.
    
    # Let's pre-compute the t-subsets for each block as a numpy array for faster indexing.
    block_tsubs_arr = [np.array(subs) for subs in block_to_tsubs]
    
    # And use a sparse representation? No, numpy dense is fine for 3876 elements.
    
    # Let's try to optimize the local search instead, as it's where the real improvement comes from.
    # The greedy phase just needs to find a valid cover.
    
    def local_search(solution_blocks, solution_set, cover_count):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Pass 1: Try to remove any redundant block
            for i in range(len(solution_blocks)):
                b_idx = solution_blocks[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break

            # Pass 2: If no removal, try to swap a block to create redundancy elsewhere
            if not improved and time.time() < deadline - 5:
                for i in range(len(solution_blocks)):
                    b_idx = solution_blocks[i]
                    critical_subsets = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    
                    if not critical_subsets:
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_blocks.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                        
                    candidate_set = set(tsub_to_blocks[critical_subsets[0]])
                    for s in critical_subsets[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                            
                    if candidate_set:
                        best_candidate = -1
                        best_score = -1.0
                        for c_idx in candidate_set:
                            if c_idx not in solution_set:
                                new_cover_gain = sum(1 for s in block_to_tsubs[c_idx] if cover_count[s] <= 1)
                                score = new_cover_gain / (1.0 + block_sizes[c_idx] * 0.1)
                                if score > best_score:
                                    best_score = score
                                    best_candidate = c_idx
                                    
                        if best_candidate != -1:
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            solution_blocks[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            improved = True
                            break

            # Pass 3: If still no improvement, try a 2-for-1 swap (replace 2 blocks with 1)
            if not improved and time.time() < deadline - 5:
                criticals = []
                for i in range(len(solution_blocks)):
                    b_idx = solution_blocks[i]
                    crits = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    criticals.append(crits)
                    
                n_pairs_to_try = 50
                for _ in range(n_pairs_to_try):
                    if time.time() > deadline - 5:
                        break
                    i, j = random.sample(range(len(solution_blocks)), 2)
                    b_i = solution_blocks[i]
                    b_j = solution_blocks[j]
                    
                    all_crits = criticals[i] + criticals[j]
                    if not all_crits:
                        continue
                        
                    candidate_set = set(tsub_to_blocks[all_crits[0]])
                    for s in all_crits[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                    
                    if candidate_set:
                        valid_candidates = [c for c in candidate_set if c not in solution_set]
                        if valid_candidates:
                            best_candidate = valid_candidates[0]
                            for s in block_to_tsubs[b_i]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b_j]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            if i < j:
                                solution_blocks.pop(j)
                                solution_blocks.pop(i)
                            else:
                                solution_blocks.pop(i)
                                solution_blocks.pop(j)
                            solution_set.remove(b_i)
                            solution_set.remove(b_j)
                            solution_set.add(best_candidate)
                            solution_blocks.append(best_candidate)
                            
                            improved = True
                            break

    best_blocks = []
    best_len = float('inf')
    
    # Use a faster greedy: pick the block that covers the most uncovered subsets.
    # To speed up, we can maintain a list of uncovered subsets and only consider blocks that cover at least one.
    
    while time.time() < deadline - 15:
        uncovered = [True] * n_t_subs
        total_uncovered = n_t_subs
        solution_blocks = []
        solution_set = set()
        
        while total_uncovered > 0:
            if time.time() > deadline - 15:
                break
            
            # Find the block with the maximum number of uncovered subsets
            max_gain = 0
            best_b_idx = -1
            
            for i in range(n_blocks):
                gain = sum(1 for s in block_to_tsubs[i] if uncovered[s])
                if gain > max_gain:
                    max_gain = gain
                    best_b_idx = i
                elif gain == max_gain and gain > 0:
                    # Tie-breaker: pick the one with lower "rarity" sum
                    rarity_sum = sum(tsub_rarity[s] for s in block_to_tsubs[i] if uncovered[s])
                    # We don't track the previous rarity sum, so this is approximate.
                    # Let's just pick randomly among the best for now.
                    if random.random() < 0.5:
                        best_b_idx = i
            
            if best_b_idx == -1:
                break
                
            solution_blocks.append(best_b_idx)
            solution_set.add(best_b_idx)
            
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = False
                    total_uncovered -= 1
        
        if total_uncovered > 0:
            continue
            
        cover_count = [0] * n_t_subs
        for b_idx in solution_blocks:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        local_search(solution_blocks, solution_set, cover_count)
        
        if len(solution_blocks) < best_len:
            best_len = len(solution_blocks)
            best_blocks = list(solution_blocks)
    
    if best_blocks and time.time() < deadline - 3:
        solution_blocks = list(best_blocks)
        solution_set = set(best_blocks)
        cover_count = [0] * n_t_subs
        for b_idx in solution_blocks:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(len(solution_blocks)):
                b_idx = solution_blocks[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break
        
        if len(solution_blocks) < best_len:
            best_len = len(solution_blocks)
            best_blocks = list(solution_blocks)
    
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return final_blocks
