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
        # Fallback if precomputation is too slow (unlikely for 19,8,5)
        return [list(k_blocks[0])] 

    # 2. Greedy Set Cover
    # Weights: 1 / (number of blocks covering the subset)
    # This is a static weight, but we can update it dynamically or just use static.
    # Static is faster.
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
        # Find best block
        # Mask out selected blocks
        current_gains = gain.copy()
        current_gains[selected_mask] = -1.0
        best_idx = np.argmax(current_gains)
        
        if gain[best_idx] <= 0:
            break
            
        selected.append(best_idx)
        selected_mask[best_idx] = True
        
        # Update uncovered and gains
        for idx in block_t_indices[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
                w = t_weights[idx]
                # Subtract weight from all blocks covering this t-subset
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= w
                        
    # 3. Remove Redundant Blocks
    # coverage_count for each t-subset
    cover_count = np.zeros(num_t, dtype=np.int32)
    for b_idx in selected:
        for s_idx in block_t_indices[b_idx]:
            cover_count[s_idx] += 1
            
    changed = True
    while changed and elapsed() < deadline - 10:
        changed = False
        new_selected = []
        for b_idx in selected:
            needed = False
            for s_idx in block_t_indices[b_idx]:
                if cover_count[s_idx] == 1:
                    needed = True
                    break
            if needed:
                new_selected.append(b_idx)
            else:
                changed = True
                for s_idx in block_t_indices[b_idx]:
                    cover_count[s_idx] -= 1
        if changed:
            selected = new_selected

    # 4. Local Search: 1-1 Swaps to enable drops
    # We want to find a swap B_out -> B_in such that B_in covers all unique subsets of B_out.
    # Then we try to drop blocks again.
    
    def try_drop_blocks():
        nonlocal selected, cover_count
        changed = True
        while changed and elapsed() < deadline - 5:
            changed = False
            new_selected = []
            for b_idx in selected:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cover_count[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cover_count[s_idx] -= 1
            if changed:
                selected = new_selected
                return True
        return False

    def score_candidate(b_in, unique_subs, b_out):
        """Score a candidate swap: prefer candidates that cover more already-covered subsets."""
        score = 0
        for s_idx in block_t_indices[b_in]:
            if cover_count[s_idx] >= 2:
                score += 1
        # Small bonus for covering subsets that b_out also covers (overlap)
        # This is a heuristic to prefer "similar" blocks
        return score

    # Local search loop
    improved = True
    while improved and elapsed() < deadline - 5:
        improved = False
        # Try to drop first
        if try_drop_blocks():
            improved = True
            continue
            
        # Try swaps - try multiple candidates per block to find better swaps
        swap_found = False
        for i in range(len(selected)):
            if elapsed() > deadline - 5:
                break
            b_out = selected[i]
            # Find unique subsets of b_out
            unique_subs = []
            for s_idx in block_t_indices[b_out]:
                if cover_count[s_idx] == 1:
                    unique_subs.append(s_idx)
            
            if not unique_subs:
                # Should have been dropped
                continue
                
            # Find candidates B_in that cover all unique_subs
            # Intersection of t_to_blocks[s] for s in unique_subs
            candidates = None
            for s_idx in unique_subs:
                if candidates is None:
                    candidates = set(t_to_blocks[s_idx])
                else:
                    candidates &= t_to_blocks[s_idx]
                    
            if not candidates:
                continue
                
            # Remove b_out itself
            candidates.discard(b_out)
            
            if not candidates:
                continue
            
            # Try up to 10 best candidates (sorted by score)
            cand_list = list(candidates)
            # Limit candidates to keep time bounded
            if len(cand_list) > 20:
                random.shuffle(cand_list)
                cand_list = cand_list[:20]
            
            # Score and sort candidates
            scored = []
            for b_in in cand_list:
                s = score_candidate(b_in, unique_subs, b_out)
                scored.append((s, b_in))
            scored.sort(reverse=True)
            
            # Try candidates in order of score
            for _, b_in in scored:
                if elapsed() > deadline - 5:
                    break
                # Perform Swap
                # Decrement counts for b_out
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                # Increment counts for b_in
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                    
                # Update selected
                selected[i] = b_in
                
                # Try to drop blocks
                if try_drop_blocks():
                    improved = True
                    swap_found = True
                    break 
                else:
                    # Revert swap
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out
            
            if swap_found:
                break

    # 5. Final polish: one more pass of drops + swaps with tighter time budget
    if elapsed() < deadline - 2:
        improved2 = True
        while improved2 and elapsed() < deadline - 2:
            improved2 = False
            if try_drop_blocks():
                improved2 = True
                continue
            # Quick swap pass
            for i in range(len(selected)):
                if elapsed() > deadline - 2:
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
                b_in = next(iter(candidates))
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                selected[i] = b_in
                if try_drop_blocks():
                    improved2 = True
                    break
                else:
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out

    # Final result
    result = [list(k_blocks[i]) for i in selected]
    return result
