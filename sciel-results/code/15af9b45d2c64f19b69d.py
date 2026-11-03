import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute the list of t-subset indices covered by each block
    block_t_indices = []
    for block in all_blocks:
        idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(idxs)
        
    # Build inverse index: for each t-subset, list of blocks that cover it
    # This is crucial for fast local search swaps
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, t_idxs in enumerate(block_t_indices):
        for t_idx in t_idxs:
            t_to_blocks[t_idx].append(b_idx)

    # 2. Initial Solution via Greedy Set Cover
    # We use a bitmask to track coverage for fast gain calculation
    n_words = (n_t + 63) // 64
    covered_mask = np.zeros(n_words, dtype=np.uint64)
    selected_blocks = []
    
    # Precompute coverage masks for all blocks for the greedy phase
    # mask[b_idx, w_idx] has bit i set if block b_idx covers t_subset (w_idx*64+i)
    coverage_masks = np.zeros((n_blocks, n_words), dtype=np.uint64)
    for b_idx, t_idxs in enumerate(block_t_indices):
        for t_idx in t_idxs:
            w = t_idx // 64
            b = t_idx % 64
            coverage_masks[b_idx, w] |= (1 << b)

    # Greedy selection
    max_greedy_steps = 200
    for _ in range(max_greedy_steps):
        if time.time() > deadline - 50.0:
            break
        if np.bitwise_count(covered_mask).sum() == n_t:
            break
            
        # Calculate gain for each block: number of new t-subsets covered
        # new_bits = coverage_masks & ~covered_mask
        # gain = sum(popcount(new_bits, axis=1))
        not_covered = ~covered_mask
        intersections = coverage_masks & not_covered
        gains = np.bitwise_count(intersections).sum(axis=1)
        
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
            
        selected_blocks.append(best_idx)
        covered_mask |= coverage_masks[best_idx]
        
    # 3. Local Search: Simulated Annealing
    # State: list of selected block indices
    # Objective: Minimize number of blocks, while maintaining validity.
    # Moves:
    # 1. Remove a redundant block (if it doesn't uncover any t-subset).
    # 2. Swap a block with another block that maintains validity.
    
    # Convert to set for fast membership check
    selected_set = set(selected_blocks)
    current_selected = list(selected_blocks)
    
    # Compute cover counts for the current solution
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in current_selected:
        cover_counts[block_t_indices[b_idx]] += 1
        
    best_selected = current_selected.copy()
    best_size = len(current_selected)
    
    # Simulated Annealing Parameters
    T = 10.0
    min_T = 0.1
    cooling = 0.99
    
    # Precompute length of t_to_blocks lists for candidate sampling
    t_to_blocks_len = np.array([len(l) for l in t_to_blocks], dtype=np.int32)
    
    rng = random.Random(42)
    
    # Local Search Loop
    while time.time() < deadline - 5.0:
        if T < min_T:
            T = 5.0 # Reheat occasionally to escape local minima? Or just reset.
            
        # Pick a random block to modify
        if not current_selected:
            break
            
        i = rng.randint(0, len(current_selected) - 1)
        b_idx = current_selected[i]
        t_idxs = block_t_indices[b_idx]
        
        # Calculate "deficit" if we remove b_idx
        # Deficit are t-subsets where cover_counts[t_idx] == 1
        # If we remove b_idx, those become 0 (uncovered)
        deficit_mask = (cover_counts == 1)
        deficit_idxs = t_idxs[deficit_mask[t_idxs]]
        
        new_size = len(current_selected)
        
        if len(deficit_idxs) == 0:
            # Safe to remove
            cover_counts[t_idxs] -= 1
            current_selected.pop(i)
            selected_set.discard(b_idx)
            new_size -= 1
            
            if new_size < best_size:
                best_size = new_size
                best_selected = current_selected.copy()
            
            T *= cooling
            continue
            
        else:
            # Need to swap b_idx with a block j that covers all deficit_idxs
            # And j must not be in selected_set
            
            # To speed up, pick the most constrained deficit (one with fewest covering blocks)
            # But for simplicity, just pick the first one or a random one?
            # Let's pick the one with the fewest candidates to increase success rate?
            # Actually, any deficit must be covered.
            # Let's pick a random deficit index to sample from.
            
            if len(deficit_idxs) > 0:
                # Optimization: If there are many deficits, it's hard to satisfy all.
                # We can pick the deficit with the smallest t_to_blocks list to constrain candidates.
                # But checking all deficits is expensive.
                # Let's just pick a random deficit.
                d_idx = rng.choice(deficit_idxs)
                candidates = t_to_blocks[d_idx]
                
                if not candidates:
                    T *= cooling
                    continue
                
                # Try a few random candidates
                # We want j such that set(deficit_idxs) subset of set(block_t_indices[j])
                # And j not in selected_set
                
                tried = 0
                max_tries = 20
                found_swap = False
                
                while tried < max_tries:
                    j_idx = rng.choice(candidates)
                    tried += 1
                    
                    if j_idx in selected_set:
                        continue
                    
                    # Check if j covers all deficit_idxs
                    # j's t-subsets are block_t_indices[j_idx]
                    # We need to check if all d in deficit_idxs are in block_t_indices[j_idx]
                    
                    # Efficient check:
                    # If len(deficit_idxs) is small, just iterate.
                    # If large, maybe precompute sets? But sets are expensive to build.
                    # Let's assume deficit_idxs is small (usually 1-5).
                    
                    valid = True
                    j_t_set = set(block_t_indices[j_idx])
                    for d in deficit_idxs:
                        if d not in j_t_set:
                            valid = False
                            break
                            
                    if valid:
                        # Perform swap
                        # Remove b_idx, add j_idx
                        cover_counts[t_idxs] -= 1
                        cover_counts[block_t_indices[j_idx]] += 1
                        
                        current_selected[i] = j_idx
                        selected_set.discard(b_idx)
                        selected_set.add(j_idx)
                        
                        # new_size remains same
                        # We don't update best here because size didn't improve
                        # But this move might enable future removals
                        found_swap = True
                        break
                        
                if found_swap:
                    # Accept swap (always, since it maintains validity and size is same)
                    # In SA, we might accept worse moves, but here "worse" is harder to define
                    # since size is constant. We just want to change state.
                    pass

        T *= cooling
        if T < min_T:
            # Reheat
            T = 5.0

    # Finalize
    final_selected = best_selected
    final_blocks = [list(all_blocks[i]) for i in final_selected]
    
    # Verify validity
    test_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in final_selected:
        test_counts[block_t_indices[b_idx]] += 1
        
    if np.any(test_counts == 0):
        # Fallback to greedy result (which was valid)
        # Re-run greedy to be safe? Or just use the initial selected_blocks?
        # The initial greedy result is stored in `selected_blocks`?
        # No, `selected_blocks` was modified? No, we used `current_selected` as a copy.
        # So `selected_blocks` still holds the greedy result.
        # But wait, we didn't make a copy of selected_blocks for greedy.
        # Let's check the code.
        # `selected_blocks` is the greedy list.
        # `current_selected` is `list(selected_blocks)`.
        # So `selected_blocks` is unchanged.
        return [list(all_blocks[i]) for i in selected_blocks]
        
    return final_blocks
