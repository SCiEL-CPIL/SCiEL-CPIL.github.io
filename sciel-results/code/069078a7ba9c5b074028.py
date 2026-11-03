import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(points, k))
    num_k = len(k_blocks)
    
    # Map each block to the indices of t-subsets it covers
    block_t_indices = []
    for block in k_blocks:
        covered = set()
        for sub in itertools.combinations(block, t):
            covered.add(t_index[sub])
        block_t_indices.append(list(covered))
        
    # Map each t-subset to the indices of blocks that cover it
    t_to_blocks = [set() for _ in range(num_t)]
    for b_idx, subs in enumerate(block_t_indices):
        for s_idx in subs:
            t_to_blocks[s_idx].add(b_idx)

    if elapsed() > deadline - 60:
        return [list(k_blocks[0])] 

    # 2. Randomized Greedy Set Cover with multiple restarts
    def randomized_greedy(seed_val):
        rng = random.Random(seed_val)
        
        # Static weights based on frequency
        t_weights = np.zeros(num_t, dtype=np.float64)
        for i in range(num_t):
            count = len(t_to_blocks[i])
            if count > 0:
                t_weights[i] = 1.0 / count
            else:
                t_weights[i] = 0.0
            
        # Initialize gain for each block
        gain = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if block_t_indices[i]:
                gain[i] = np.sum(t_weights[block_t_indices[i]])
                
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        
        while uncovered_count > 0 and elapsed() < deadline - 40:
            # Find best block with random tie-breaking
            current_gains = gain.copy()
            current_gains[selected_mask] = -1.0
            best_idx = np.argmax(current_gains)
            
            if gain[best_idx] <= 0:
                break
                
            # Random tie-breaking: among blocks with similar gain, pick randomly
            threshold = gain[best_idx] * 0.99
            candidates = np.where((current_gains >= threshold) & (~selected_mask))[0]
            if len(candidates) > 1:
                best_idx = candidates[rng.randrange(len(candidates))]
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered and gains
            for idx in block_t_indices[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w
                        
        return selected

    # Run multiple greedy restarts
    best_selected = None
    best_size = float('inf')
    num_restarts = 8
    restart_time_budget = deadline - 80  # Leave time for local search
    
    for r in range(num_restarts):
        if elapsed() > restart_time_budget:
            break
        selected = randomized_greedy(r * 12345 + 67890)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]

    if best_selected is None:
        best_selected = randomized_greedy(0)
        best_size = len(best_selected)

    # 3. Remove Redundant Blocks
    selected = best_selected[:]
    
    def compute_cover_count(sel):
        cc = np.zeros(num_t, dtype=np.int32)
        for b_idx in sel:
            for s_idx in block_t_indices[b_idx]:
                cc[s_idx] += 1
        return cc

    cover_count = compute_cover_count(selected)
    
    def try_drop_blocks(sel, cc):
        changed = True
        while changed and elapsed() < deadline - 10:
            changed = False
            new_selected = []
            for b_idx in sel:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cc[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cc[s_idx] -= 1
            if changed:
                sel = new_selected
        return sel, cc

    selected, cover_count = try_drop_blocks(selected, cover_count)

    # 4. Local Search: 1-1 Swaps and 2-1 Swaps
    def try_drop_blocks_local(sel, cc):
        changed = True
        while changed and elapsed() < deadline - 5:
            changed = False
            new_selected = []
            for b_idx in sel:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cc[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cc[s_idx] -= 1
            if changed:
                sel = new_selected
        return sel, cc

    improved = True
    while improved and elapsed() < deadline - 5:
        improved = False
        
        # Try to drop first
        selected, cover_count = try_drop_blocks_local(selected, cover_count)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
            improved = True
            continue
            
        # Try 1-1 swaps
        for i in range(len(selected)):
            if elapsed() > deadline - 5:
                break
            b_out = selected[i]
            unique_subs = [s_idx for s_idx in block_t_indices[b_out] if cover_count[s_idx] == 1]
            
            if not unique_subs:
                continue
                
            candidates = None
            for s_idx in unique_subs:
                if candidates is None:
                    candidates = set(t_to_blocks[s_idx])
                else:
                    candidates &= t_to_blocks[s_idx]
                    
            if not candidates:
                continue
                
            candidates.discard(b_out)
            if not candidates:
                continue
                
            # Try each candidate (limited)
            candidate_list = list(candidates)
            rng = random.Random(i * 1000 + len(selected))
            rng.shuffle(candidate_list)
            
            for b_in in candidate_list[:5]:  # Limit to 5 candidates per position
                # Perform swap
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                    
                selected[i] = b_in
                
                new_sel, new_cc = try_drop_blocks_local(selected, cover_count)
                
                if len(new_sel) < len(selected):
                    selected, cover_count = new_sel, new_cc
                    best_size = len(selected)
                    best_selected = selected[:]
                    improved = True
                    break
                else:
                    # Revert
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out

    # Final result
    result = [list(k_blocks[i]) for i in best_selected]
    return result
