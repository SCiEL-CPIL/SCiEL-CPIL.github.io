import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    # 1. Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # 3. Precompute coverage mask for each k-subset and list of t-subset indices
    ksub_masks = [0] * n_ksubs
    block_tsub_indices = [[] for _ in range(n_ksubs)]
    
    # Also compute total frequency of each t-subset for weighted greedy
    tsub_freq = [0] * n_tsubs

    for j, ks in enumerate(all_ksubs):
        mask = 0
        indices = []
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
            indices.append(idx)
            tsub_freq[idx] += 1
        ksub_masks[j] = mask
        block_tsub_indices[j] = indices

    # Precompute weights for weighted greedy: w_s = 1 / freq_s
    # If freq is 0 (shouldn't happen), weight is 0
    weights = [1.0 / f if f > 0 else 0.0 for f in tsub_freq]
    
    # 4. Greedy Selection Function
    def greedy_select(use_weights):
        uncovered_mask = (1 << n_tsubs) - 1
        remaining = n_tsubs
        blocks = []
        used_blocks = set()
        
        while remaining > 0:
            if elapsed() > deadline - 5.0: # Leave time for refinement
                break
            
            best_j = -1
            best_gain = -1.0
            
            # Scan all candidates
            for j in range(n_ksubs):
                if j in used_blocks:
                    continue
                
                common = uncovered_mask & ksub_masks[j]
                if common == 0:
                    continue
                
                if use_weights:
                    # Calculate weighted gain
                    gain = 0.0
                    # Iterate over the t-subsets in this block
                    # Only those that are in 'common'
                    for idx in block_tsub_indices[j]:
                        if (common >> idx) & 1:
                            gain += weights[idx]
                else:
                    gain = common.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            
            if best_j == -1 or best_gain <= 0:
                break
            
            blocks.append(all_ksubs[best_j])
            used_blocks.add(best_j)
            
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= (uncovered_mask.bit_count() if not use_weights else 0) # Just for loop condition, bit_count is slow if called every time? 
            # Actually, we can just check if uncovered_mask == 0
            # But 'remaining' is faster.
            # For weighted, we don't track 'remaining' accurately in the simple way.
            # Let's just check if uncovered_mask == 0
            if uncovered_mask == 0:
                break

        return blocks

    # 5. Run both strategies
    blocks_std = greedy_select(False)
    blocks_wtd = greedy_select(True)
    
    # Choose the better one (fewer blocks)
    if len(blocks_std) <= len(blocks_wtd):
        blocks = blocks_std
    else:
        blocks = blocks_wtd
        
    # 6. Refinement: Remove redundant blocks
    if elapsed() < deadline - 2.0 and len(blocks) > 0:
        # Map blocks back to indices
        selected_indices = []
        # Create a map from tuple to index for faster lookup
        ksub_to_idx = {ks: i for i, ks in enumerate(all_ksubs)}
        
        for b in blocks:
            idx = ksub_to_idx.get(tuple(b), -1)
            if idx != -1:
                selected_indices.append(idx)
        
        active_indices = selected_indices
        
        changed = True
        while changed and elapsed() < deadline - 1.0:
            changed = False
            new_active = []
            for i_idx, bi in enumerate(active_indices):
                # Check if bi is redundant
                # Calculate mask of all other active blocks
                other_mask = 0
                for j_idx, bj in enumerate(active_indices):
                    if i_idx != j_idx:
                        other_mask |= ksub_masks[bj]
                
                # If all bits in bi are in other_mask, bi is redundant
                if (ksub_masks[bi] & ~other_mask) == 0:
                    changed = True
                else:
                    new_active.append(bi)
            
            active_indices = new_active
        
        if active_indices:
            final_blocks = [list(all_ksubs[idx]) for idx in active_indices]
            # Verify validity
            final_mask = 0
            for idx in active_indices:
                final_mask |= ksub_masks[idx]
            if final_mask == (1 << n_tsubs) - 1:
                blocks = final_blocks

    return blocks
