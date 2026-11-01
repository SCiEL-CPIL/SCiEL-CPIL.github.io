import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage: which t-subsets does each block cover?
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
        
    # Precompute inverse: which blocks cover each t-subset?
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Precompute scarcity weights for greedy bias
    # weight[b] = sum(1 / degree(t) for t in block)
    tsub_degrees = [len(lst) for lst in tsub_to_blocks]
    block_scarcity = np.zeros(num_blocks, dtype=np.float64)
    for b_idx, tsubs in enumerate(block_to_tsubs):
        w = 0.0
        for s_idx in tsubs:
            if tsub_degrees[s_idx] > 0:
                w += 1.0 / tsub_degrees[s_idx]
        block_scarcity[b_idx] = w

    # Helper: Weighted Greedy Construction
    def greedy_construction(randomize_ties=False, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = np.ones(num_tsubs, dtype=bool)
        gains = np.full(num_blocks, len(block_to_tsubs[0]), dtype=np.int32)
        
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None # Timeout
            
            # Calculate score: gain * large_factor + scarcity
            # We only need to consider blocks with gain > 0
            # To be fast, we can just argmax gains, then break ties with scarcity
            # But gains change.
            
            # Efficient max gain search
            # Since gains are small ints, we can find max gain and then pick among those with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max_gain
            # This can be done with np.where(gains == max_gain)
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                b = candidates[0]
            else:
                # Tie-breaking: Pick block with highest scarcity among candidates
                # Or randomize
                if randomize_ties:
                    # With some probability, pick random from candidates
                    # With some probability, pick highest scarcity
                    if random.random() < 0.5:
                        b = random.choice(candidates)
                    else:
                        # Find max scarcity among candidates
                        cand_scarcity = block_scarcity[candidates]
                        b = candidates[np.argmax(cand_scarcity)]
                else:
                    # Deterministic: Highest scarcity
                    cand_scarcity = block_scarcity[candidates]
                    b = candidates[np.argmax(cand_scarcity)]
            
            selected_block_indices.append(b)
            
            # Update uncovered and gains
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        if uncovered.any():
            return None # Failed to cover
            
        return selected_block_indices

    # Helper: Local Optimization
    def local_optimize(blocks):
        if not blocks:
            return []
            
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step A: Redundancy Removal
            redundant_blocks = []
            for b in selected_list:
                is_red = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        is_red = False
                        break
                if is_red:
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue # Restart
            
            # Step B: 1-for-1 Swaps
            # Calculate unique t-subsets for each block
            unique_tsubs_map = {}
            unique_counts = {}
            for b in selected_list:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            if not unique_counts:
                break
                
            # Sort blocks by unique count to try removing "easy" ones first
            sorted_blocks = sorted(selected_list, key=lambda x: unique_counts[x])
            
            swap_made = False
            for b_out in sorted_blocks:
                if time.time() > deadline:
                    break
                s_uniq = unique_tsubs_map[b_out]
                if not s_uniq:
                    continue # Should have been removed as redundant
                
                # Find replacement
                # Candidates must cover all s_uniq
                # Start with candidates of the first unique subset
                first_s = s_uniq[0]
                candidates = tsub_to_blocks_np[first_s]
                
                best_candidate = None
                # We want a candidate that covers all s_uniq and is not in selected_set
                # Ideally, we want a candidate that maximizes "extra" coverage? No, just any valid one.
                # To improve, we might prefer one that allows further reductions?
                # For now, just find any.
                
                for c in candidates:
                    if c in selected_set:
                        continue
                    c_set = block_to_tsubs_set[c]
                    covers_all = True
                    for s in s_uniq:
                        if s not in c_set:
                            covers_all = False
                            break
                    if covers_all:
                        best_candidate = c
                        break # Take first one found? Or random?
                        # Taking first one found might be biased. Let's collect a few and pick random.
                        # But for speed, first found is okay.
                
                if best_candidate is not None:
                    # Perform swap
                    selected_list.remove(b_out)
                    selected_set.remove(b_out)
                    selected_list.append(best_candidate)
                    selected_set.add(best_candidate)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[best_candidate]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart loop after one swap
            
            if not swap_made:
                # Step C: 2-for-1 Swaps (Limited)
                # Try to find two blocks whose union of unique subsets is covered by a third block
                # This is O(N^2). Limit to a few iterations or random pairs.
                
                # We'll try a limited number of random pairs to avoid timeout
                pairs_tried = 0
                max_pairs = 1000
                while pairs_tried < max_pairs and time.time() < deadline:
                    if len(selected_list) < 2:
                        break
                    b1, b2 = random.sample(selected_list, 2)
                    pairs_tried += 1
                    
                    # Calculate union of unique subsets
                    # Unique subsets of b1: cover_count[s] == 1 and s in b1
                    # Unique subsets of b2: cover_count[s] == 1 and s in b2
                    # Note: If a subset is covered by both b1 and b2, and no one else, it's unique to the pair?
                    # No, if cover_count[s] == 2, and b1, b2 are the only ones, then removing both leaves it uncovered.
                    # So we need to consider subsets where cover_count[s] <= 2?
                    # Actually, if we remove b1 and b2, a subset s is uncovered if cover_count[s] - (1 if s in b1 else 0) - (1 if s in b2 else 0) == 0.
                    
                    required_subsets = []
                    for s in block_to_tsubs[b1]:
                        cnt = cover_count[s]
                        if s in block_to_tsubs[b2]:
                            cnt -= 1
                        if cnt == 0:
                            required_subsets.append(s)
                    for s in block_to_tsubs[b2]:
                        if s in block_to_tsubs[b1]:
                            continue # Already processed
                        if cover_count[s] == 0:
                            required_subsets.append(s)
                    
                    if not required_subsets:
                        # Both are redundant? Should have been caught in Step A.
                        continue
                    
                    # Find a block c that covers all required_subsets
                    # c must not be b1 or b2
                    first_req = required_subsets[0]
                    candidates = tsub_to_blocks_np[first_req]
                    
                    found = False
                    for c in candidates:
                        if c == b1 or c == b2 or c in selected_set:
                            continue
                        c_set = block_to_tsubs_set[c]
                        covers_all = True
                        for s in required_subsets:
                            if s not in c_set:
                                covers_all = False
                                break
                        if covers_all:
                            # Perform 2-for-1 swap
                            selected_list.remove(b1)
                            selected_list.remove(b2)
                            selected_set.remove(b1)
                            selected_set.remove(b2)
                            selected_list.append(c)
                            selected_set.add(c)
                            
                            for s in block_to_tsubs[b1]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b2]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[c]:
                                cover_count[s] += 1
                            
                            found = True
                            improved = True
                            break
                            
                    if found:
                        break # Restart main loop
                
                if not improved:
                    break
                    
        return selected_list

    best_solution = None
    best_len = float('inf')
    
    # Multi-start strategy
    # Run greedy + local search multiple times
    
    # First run deterministic
    blocks = greedy_construction(randomize_ties=False, random_seed=42)
    if blocks is not None:
        optimized_blocks = local_optimize(blocks)
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
    
    # Subsequent runs with randomization
    for i in range(1, 30):
        if time.time() > deadline:
            break
            
        blocks = greedy_construction(randomize_ties=True, random_seed=42 + i)
        
        if blocks is None:
            continue
            
        optimized_blocks = local_optimize(blocks)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
            
    # If we have time, do ILS on the best solution
    if best_solution is not None and time.time() < deadline - 10:
        current_best = list(best_solution)
        
        # Perturb and Repair
        for _ in range(5): # 5 ILS iterations
            if time.time() > deadline - 5:
                break
            
            # Perturb: Remove 5 random blocks
            perturbed = list(current_best)
            if len(perturbed) > 10:
                remove_indices = random.sample(range(len(perturbed)), 5)
                remove_indices.sort(reverse=True)
                for idx in remove_indices:
                    perturbed.pop(idx)
            
            # Repair: Greedy to cover gaps
            # We need to know what is uncovered by 'perturbed'
            cover_count = np.zeros(num_tsubs, dtype=np.int32)
            for b in perturbed:
                for s in block_to_tsubs[b]:
                    cover_count[s] += 1
            
            uncovered_mask = cover_count == 0
            
            # Run a simple greedy to fill gaps
            # We can reuse the logic from greedy_construction but start with 'perturbed'
            # For simplicity, we'll just add blocks greedily
            selected_set = set(perturbed)
            
            # Calculate gains for all blocks relative to current uncovered
            gains = np.zeros(num_blocks, dtype=np.int32)
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                gain = 0
                for s in block_to_tsubs[b_idx]:
                    if uncovered_mask[s]:
                        gain += 1
                gains[b_idx] = gain
            
            while uncovered_mask.any():
                if time.time() > deadline - 2:
                    break
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                # Pick one with highest scarcity
                cand_scarcity = block_scarcity[candidates]
                b = candidates[np.argmax(cand_scarcity)]
                
                selected_set.add(b)
                for s in block_to_tsubs[b]:
                    if uncovered_mask[s]:
                        uncovered_mask[s] = False
                        gains[tsub_to_blocks_np[s]] -= 1
            
            # Local Optimize the result
            result_blocks = list(selected_set)
            optimized = local_optimize(result_blocks)
            
            if len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized
                current_best = optimized

    if best_solution is None:
        return []
        
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
