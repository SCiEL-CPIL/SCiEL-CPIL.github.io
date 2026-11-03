"""Covering design C(v,k,t) via fast bitmask greedy + redundancy removal."""
import itertools
import time


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # Safety margin under 120s limit

    # 1. Precomputation
    # Generate all k-subsets (blocks)
    all_blocks = [tuple(c) for c in itertools.combinations(range(v), k)]
    
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # For each block, compute the bitmask of t-subsets it covers
    # Using Python integers as bitsets (arbitrary precision)
    block_masks = [0] * len(all_blocks)
    for i, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_index[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    # 2. Greedy Set Cover
    # uncovered_mask starts with all bits set (all subsets uncovered)
    # We want to clear bits as we cover subsets.
    # Actually, it's easier to track 'covered' and check 'uncovered' = all_mask ^ covered
    all_mask = (1 << num_subsets) - 1
    covered = 0
    selected_blocks = []
    
    # Optimization: We can filter blocks that are subsets of others? No, all are size k.
    # We can remove duplicate blocks? itertools.combinations gives unique.
    
    while covered != all_mask:
        if time.time() > deadline:
            break
        
        best_block_idx = -1
        best_gain = 0
        
        # Find the block that covers the most *uncovered* subsets
        # Iterate over all blocks. 75k blocks.
        # bit_count is fast.
        for i in range(len(all_blocks)):
            # Gain is number of bits in block_masks[i] that are NOT in covered
            # i.e., bits in block_masks[i] that are in (all_mask ^ covered)
            gain = (block_masks[i] & (all_mask ^ covered)).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                # Early termination if we found a block that covers everything left?
                # Unlikely, but if gain == (all_mask ^ covered).bit_count():
                #   break
                # But we need to be careful.
        
        if best_block_idx == -1 or best_gain == 0:
            # No improvement possible? Should not happen if covered != all_mask
            # and there are blocks covering remaining subsets.
            break
            
        selected_blocks.append(all_blocks[best_block_idx])
        covered |= block_masks[best_block_idx]

    # 3. Redundancy Removal (Local Search)
    # We have a list of selected block indices (implicitly, we can re-derive or store)
    # Let's store indices to make it easier to work with.
    # Actually, the loop above didn't store indices. Let's refactor to store indices.
    
    # Refactored Greedy to store indices
    covered = 0
    selected_indices = []
    
    while covered != all_mask:
        if time.time() > deadline:
            break
        
        best_block_idx = -1
        best_gain = 0
        
        for i in range(len(all_blocks)):
            gain = (block_masks[i] & (all_mask ^ covered)).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        
        if best_block_idx == -1 or best_gain == 0:
            break
            
        selected_indices.append(best_block_idx)
        covered |= block_masks[best_block_idx]

    # Redundancy Removal Pass
    # Count how many times each subset is covered by the selected blocks
    # subset_counts[j] = number of selected blocks covering subset j
    
    # Efficiently update counts:
    # Initialize counts to 0
    # For each selected block, iterate its subsets and increment count.
    # Then, a block is redundant if all its subsets have count > 1.
    # Remove it, decrement counts for its subsets.
    # Repeat until no more removals.
    
    # To make this fast, we can precompute the list of subset indices for each block.
    block_subsets = []
    for i, block in enumerate(all_blocks):
        subs = [sub_index[t_sub] for t_sub in itertools.combinations(block, t)]
        block_subsets.append(subs)

    # We only care about selected blocks, but block_subsets is for all.
    # Let's work with a set of selected block indices.
    selected_set = set(selected_indices)
    
    # Compute initial counts
    # Use a list of ints for counts, size num_subsets
    subset_counts = [0] * num_subsets
    for idx in selected_set:
        for s_idx in block_subsets[idx]:
            subset_counts[s_idx] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find a redundant block
        # A block is redundant if for all s in block_subsets[idx], subset_counts[s] > 1
        # We can check this quickly.
        redundant_idx = -1
        for idx in selected_set:
            is_redundant = True
            for s_idx in block_subsets[idx]:
                if subset_counts[s_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_idx = idx
                break
        
        if redundant_idx != -1:
            # Remove it
            selected_set.remove(redundant_idx)
            # Decrement counts
            for s_idx in block_subsets[redundant_idx]:
                subset_counts[s_idx] -= 1
            changed = True
            # Restart loop to find more redundancies (as counts changed)
            # Actually, we can just continue the loop, but we need to re-evaluate all blocks?
            # Removing a block only decreases counts. It cannot make a previously non-redundant block redundant
            # (because counts go down, so if count was 1, it stays 1 or becomes 0? No, if count was 1, it becomes 0? 
            # Wait. If a block was NOT redundant, it had at least one subset with count 1.
            # If we remove another block, that count might drop to 0? No, we only remove redundant blocks.
            # A redundant block has all counts > 1. So removing it decreases counts by 1.
            # So counts stay >= 1.
            # Therefore, a block that was NOT redundant (had a count 1) remains NOT redundant (count is still 1).
            # So we don't need to re-check blocks that were already known to be non-redundant?
            # Actually, the set of redundant blocks can only grow or stay same?
            # If block A was redundant, removing block B (redundant) might make A still redundant?
            # Yes.
            # So we can just iterate through the remaining set.
            # But wait, if we remove a block, we don't need to re-check the ones we already checked in this pass?
            # We just found one. We remove it. We continue checking from where we left off?
            # Or restart? Restarting is safer and simple. With ~300 blocks, it's fast.
            pass

    # 4. Local Search: 1-for-1 swaps
    if time.time() < deadline - 15:
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for idx in list(selected_set):
                lost = []
                for s_idx in block_subsets[idx]:
                    if subset_counts[s_idx] == 1:
                        lost.append(s_idx)

                if not lost:
                    selected_set.remove(idx)
                    for s_idx in block_subsets[idx]:
                        subset_counts[s_idx] -= 1
                    improved = True
                    break

                for j in range(len(all_blocks)):
                    if j in selected_set:
                        continue
                    covers_all = True
                    for s_idx in lost:
                        if not (block_masks[j] >> s_idx) & 1:
                            covers_all = False
                            break
                    if covers_all:
                        selected_set.remove(idx)
                        for s_idx in block_subsets[idx]:
                            subset_counts[s_idx] -= 1
                        selected_set.add(j)
                        for s_idx in block_subsets[j]:
                            subset_counts[s_idx] += 1
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # 5. Second redundancy removal pass after local search
    changed = True
    while changed and time.time() < deadline:
        changed = False
        redundant_idx = -1
        for idx in selected_set:
            is_redundant = True
            for s_idx in block_subsets[idx]:
                if subset_counts[s_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_idx = idx
                break
        
        if redundant_idx != -1:
            selected_set.remove(redundant_idx)
            for s_idx in block_subsets[redundant_idx]:
                subset_counts[s_idx] -= 1
            changed = True

    # Final check: ensure all subsets are still covered
    final_covered = 0
    for idx in selected_set:
        final_covered |= block_masks[idx]
        
    if final_covered != all_mask:
        # Fallback to greedy without local search
        covered = 0
        selected_indices = []
        while covered != all_mask:
            if time.time() > deadline:
                break
            uncovered_mask = all_mask ^ covered
            best_block_idx = -1
            best_gain = 0
            for i in range(len(all_blocks)):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            if best_block_idx == -1:
                break
            selected_indices.append(best_block_idx)
            covered |= block_masks[best_block_idx]
        selected_set = set(selected_indices)

    # Convert indices back to blocks
    result_blocks = [all_blocks[i] for i in selected_set]
    
    return result_blocks
