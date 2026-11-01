"""Covering design C(19,8,5) - Global Bitmask Greedy with Local Search."""
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator kills at 150s)

    # 1. Generate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)
    
    # Map subset tuple to index
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. Precompute bitmask for each block
    block_masks = [0] * n_blocks
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask

    # Precompute for each block: list of subset indices it covers (for fast gap computation)
    block_cover_lists = []
    for b_idx in range(n_blocks):
        mask = block_masks[b_idx]
        covers = []
        m = mask
        while m:
            idx = (m & -m).bit_length() - 1
            covers.append(idx)
            m &= m - 1
        block_cover_lists.append(covers)

    # Precompute: for each subset index, which blocks cover it (for 1-exchange)
    # subset_to_blocks[s] = list of block indices that cover subset s
    subset_to_blocks = [[] for _ in range(n_subs)]
    for b_idx in range(n_blocks):
        for s in block_cover_lists[b_idx]:
            subset_to_blocks[s].append(b_idx)

    best_blocks = None
    best_count = float('inf')

    rng = random.Random(42)
    max_restarts = 300  # More restarts due to faster local search

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_subs) - 1
        selected_blocks = []
        
        active_blocks = list(range(n_blocks))
        
        while uncovered_mask:
            if time.time() > deadline:
                break

            best_gain = 0
            best_block_candidates = []

            new_active_blocks = []
            for b_idx in active_blocks:
                mask = block_masks[b_idx]
                if not (mask & uncovered_mask):
                    continue
                gain = (mask & uncovered_mask).bit_count()
                if gain > 0:
                    new_active_blocks.append(b_idx)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_candidates = [b_idx]
                    elif gain == best_gain:
                        best_block_candidates.append(b_idx)

            if not best_block_candidates:
                break

            chosen_block = rng.choice(best_block_candidates)
            selected_blocks.append(chosen_block)
            uncovered_mask &= ~block_masks[chosen_block]
            active_blocks = new_active_blocks

            if not active_blocks:
                break

        if uncovered_mask == 0:
            # Local Search: Delete redundant blocks
            cover_counts = np.zeros(n_subs, dtype=np.int32)
            for b_idx in selected_blocks:
                for s in block_cover_lists[b_idx]:
                    cover_counts[s] += 1
            
            changed = True
            while changed:
                changed = False
                for i in range(len(selected_blocks)):
                    b_idx = selected_blocks[i]
                    mask = block_masks[b_idx]
                    is_redundant = True
                    for s in block_cover_lists[b_idx]:
                        if cover_counts[s] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for s in block_cover_lists[b_idx]:
                            cover_counts[s] -= 1
                        selected_blocks.pop(i)
                        changed = True
                        break

            # 1-exchange: remove one block, re-cover gap with one block
            improved_1x = True
            while improved_1x and time.time() < deadline:
                improved_1x = False
                n_sel = len(selected_blocks)
                if n_sel < 2:
                    break
                # Try random single deletions
                n_tries = min(100, n_sel)
                for _ in range(n_tries):
                    if time.time() > deadline:
                        break
                    i = rng.randrange(n_sel)
                    b_idx = selected_blocks[i]
                    # Compute gap: subsets where cover_count == 1 and in this block
                    gap_mask = 0
                    for s in block_cover_lists[b_idx]:
                        if cover_counts[s] == 1:
                            gap_mask |= (1 << s)
                    if gap_mask == 0:
                        # Block is redundant
                        for s in block_cover_lists[b_idx]:
                            cover_counts[s] -= 1
                        selected_blocks.pop(i)
                        improved_1x = True
                        break
                    # Try to find a block that covers all of gap_mask
                    # Only try if gap is small enough
                    if gap_mask.bit_count() > 20:
                        continue
                    # Find candidate blocks that could cover the gap
                    # A block covers gap if (block_mask & gap_mask) == gap_mask
                    # Try a limited number of random blocks
                    for attempt in range(50):
                        cand = rng.randrange(n_blocks)
                        if cand == b_idx:
                            continue
                        if (block_masks[cand] & gap_mask) == gap_mask:
                            # Check if cand is already in selected_blocks
                            if cand in selected_blocks:
                                continue
                            # Perform the swap
                            for s in block_cover_lists[b_idx]:
                                cover_counts[s] -= 1
                            for s in block_cover_lists[cand]:
                                cover_counts[s] += 1
                            selected_blocks[i] = cand
                            improved_1x = True
                            break
                    if improved_1x:
                        break

            # 2-exchange: remove two blocks, re-cover gap with one block
            if len(selected_blocks) >= 3:
                improved_2x = True
                while improved_2x and time.time() < deadline:
                    improved_2x = False
                    n_sel = len(selected_blocks)
                    n_pairs = min(150, n_sel * (n_sel - 1) // 2)
                    for _ in range(n_pairs):
                        if time.time() > deadline:
                            break
                        i, j = rng.sample(range(n_sel), 2)
                        bi = selected_blocks[i]
                        bj = selected_blocks[j]
                        # Compute gap after removing both i and j
                        # A subset s is in gap iff:
                        #   s in bi and cover_counts[s] == 1 (only bi covers it)
                        #   s in bj and cover_counts[s] == 1 (only bj covers it)  
                        #   s in both bi and bj and cover_counts[s] == 2
                        gap_mask = 0
                        for s in block_cover_lists[bi]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        for s in block_cover_lists[bj]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        # Fix: subsets in both bi and bj with count==2
                        # The above double-counts: if s in both and count==2, 
                        # first loop adds it (count==1? no, count==2), second loop adds it (count==1? no)
                        # Wait: if s in both bi and bj, and only those two cover it, count==2.
                        # First loop: cover_counts[s]==1? No. Second loop: cover_counts[s]==1? No.
                        # So we miss it. Need to handle intersection separately.
                        # Actually, let me recompute properly.
                        # Remove the previous gap_mask and redo:
                        gap_mask = 0
                        for s in block_cover_lists[bi]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        for s in block_cover_lists[bj]:
                            if cover_counts[s] == 1:
                                gap_mask |= (1 << s)
                        # Now add subsets in intersection with count==2
                        for s in block_cover_lists[bi]:
                            if s in block_cover_lists[bj]:  # This is O(n^2), bad
                                pass
                        # Better: use bitmask intersection
                        mi = block_masks[bi]
                        mj = block_masks[bj]
                        inter = mi & mj
                        m = inter
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_counts[idx] == 2:
                                gap_mask |= (1 << idx)
                            m &= m - 1

                        if gap_mask == 0:
                            # Both redundant together
                            for s in block_cover_lists[bi]:
                                cover_counts[s] -= 1
                            for s in block_cover_lists[bj]:
                                cover_counts[s] -= 1
                            # Remove j first (higher index) then i
                            if j > i:
                                selected_blocks.pop(j)
                                selected_blocks.pop(i)
                            else:
                                selected_blocks.pop(i)
                                selected_blocks.pop(j)
                            improved_2x = True
                            break

                        # Try to re-cover gap with one new block
                        if gap_mask.bit_count() > 25:
                            continue
                        for attempt in range(50):
                            cand = rng.randrange(n_blocks)
                            if cand == bi or cand == bj:
                                continue
                            if cand in selected_blocks:
                                continue
                            if (block_masks[cand] & gap_mask) == gap_mask:
                                # Perform swap: remove bi, bj, add cand
                                for s in block_cover_lists[bi]:
                                    cover_counts[s] -= 1
                                for s in block_cover_lists[bj]:
                                    cover_counts[s] -= 1
                                for s in block_cover_lists[cand]:
                                    cover_counts[s] += 1
                                # Remove bi and bj, add cand
                                # Remove higher index first
                                if j > i:
                                    selected_blocks.pop(j)
                                    selected_blocks.pop(i)
                                else:
                                    selected_blocks.pop(i)
                                    selected_blocks.pop(j)
                                selected_blocks.append(cand)
                                improved_2x = True
                                break
                        if improved_2x:
                            break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [list(all_blocks[i]) for i in selected_blocks]
                if best_count <= 335:
                    break

    if best_blocks is None:
        return []
        
    return best_blocks
