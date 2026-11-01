import itertools
import time
import heapq
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100  # 20 second margin
    
    # 1. Precomputation
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map subset to index for faster lookup? 
    # Or just use the tuple as key in a dict.
    
    # block_subs[i]: list of t-subsets covered by block i
    # subset_blocks[s]: list of block indices covering subset s
    
    block_subs = [None] * num_blocks
    subset_blocks = {s: [] for s in all_t_subsets}
    
    for i, b in enumerate(all_blocks):
        subs = list(itertools.combinations(b, t))
        block_subs[i] = subs
        for s in subs:
            subset_blocks[s].append(i)
            
    # 2. Greedy Set Cover (Dual Approach)
    # uncovered_count[i] = number of subsets in block i that are currently uncovered
    uncovered_count = [0] * num_blocks
    # Initially, all subsets are uncovered.
    # For each block, count is C(k, t)
    init_count = len(block_subs[0])
    for i in range(num_blocks):
        uncovered_count[i] = init_count
        
    # Heap: (-count, index)
    heap = [(-uncovered_count[i], i) for i in range(num_blocks)]
    heapq.heapify(heap)
    
    selected_blocks = set()
    uncovered_subsets = set(all_t_subsets)
    
    while uncovered_subsets:
        if time.time() > deadline:
            break
            
        # Pop best candidate
        neg_count, idx = heapq.heappop(heap)
        count = -neg_count
        
        # Lazy deletion: if count is 0 or already selected, skip
        # Note: The count in heap might be stale if we updated it later.
        # We must check if this is still the valid max.
        # Actually, we just need to check if it's still "good enough" or if we should skip.
        # If uncovered_count[idx] != count, it's stale.
        if idx in selected_blocks:
            continue
        if uncovered_count[idx] != count:
            # Stale entry, re-push with current count? Or just skip?
            # Skipping is fine if there are other valid entries.
            # But if this was the max, and it's stale (lowered), we need the next one.
            # If it's stale (increased? no, counts only decrease), then it's fine.
            # Counts only decrease. So heap entries are always >= current count?
            # No, we push new entries when counts decrease.
            # So the heap contains (old_count, idx) and (new_count, idx).
            # The max will be the old_count if we don't clean up.
            # But old_count > new_count. So we pop the old one, see it's stale, and skip.
            # Then we pop the new one.
            continue
            
        if count == 0:
            # No more uncovered subsets can be covered by any block?
            # This implies uncovered_subsets is empty or we have a bug.
            break
            
        # Select block idx
        selected_blocks.add(idx)
        
        # Update uncovered_subsets and counts for blocks covering the subsets in this block
        for s in block_subs[idx]:
            if s in uncovered_subsets:
                uncovered_subsets.discard(s)
                # Update all blocks that cover s
                for b_idx in subset_blocks[s]:
                    if b_idx not in selected_blocks:
                        uncovered_count[b_idx] -= 1
                        heapq.heappush(heap, (-uncovered_count[b_idx], b_idx))
                        
    # 3. Pruning
    if time.time() < deadline - 5:
        # Convert selected_blocks to list
        sel_list = list(selected_blocks)
        
        # Calculate cover counts for all subsets
        cover_count = {s: 0 for s in all_t_subsets}
        for idx in sel_list:
            for s in block_subs[idx]:
                cover_count[s] += 1
                
        # Iterative redundancy removal
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_sel = []
            for idx in sel_list:
                redundant = True
                for s in block_subs[idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subs[idx]:
                        cover_count[s] -= 1
                    changed = True
                else:
                    new_sel.append(idx)
            sel_list = new_sel
            
        selected_blocks = set(sel_list)

    return [list(all_blocks[idx]) for idx in selected_blocks]
