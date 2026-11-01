import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # Helper: Greedy Set Cover with optional randomization
    def greedy_init(randomize=False):
        uncovered_mask = (1 << n_tsubs) - 1
        remaining = n_tsubs
        blocks = []
        used_blocks = set()
        
        while remaining > 0:
            if time.time() > deadline:
                break

            best_j = -1
            best_gain = 0
            candidates = []

            for j in range(n_ksubs):
                if j in used_blocks:
                    continue
                
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    candidates = [j]
                elif gain == best_gain and gain > 0:
                    candidates.append(j)

            if best_j == -1 or best_gain == 0:
                break

            if randomize and len(candidates) > 1:
                best_j = random.choice(candidates)

            blocks.append(list(all_ksubs[best_j]))
            used_blocks.add(best_j)
            
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= best_gain

        return blocks, used_blocks

    # Helper: Local Search / Polish
    def local_search(initial_blocks, initial_indices):
        if time.time() > deadline - 5.0:
            return initial_blocks, initial_indices

        active_indices = set(initial_indices)
        
        # Map block list to indices for quick lookup if needed, but we have initial_indices
        # Ensure consistency
        # initial_indices should correspond to initial_blocks
        
        # Helper to get unique coverage of a block in the active set
        # Optimization: Precompute total mask of active set? 
        # unique_mask(b) = mask(b) & ~(OR of masks of all other active blocks)
        # OR of all other = (Total Mask) ^ mask(b) if mask(b) is subset of Total Mask (which it is)
        # So unique_mask(b) = mask(b) & ~(Total Mask ^ mask(b)) = mask(b) & (Total Mask & ~mask(b)) ? No.
        # Let S = Total Mask.
        # S_other = S \ mask(b)
        # unique(b) = mask(b) & ~S_other = mask(b) & ~(S \ mask(b)) = mask(b) & (mask(b) | ~S) ?
        # Actually: ~S_other = ~(S & ~mask(b)) = ~S | mask(b)
        # unique(b) = mask(b) & (~S | mask(b)) = (mask(b) & ~S) | (mask(b) & mask(b)) = (mask(b) & ~S) | mask(b) = mask(b).
        # Wait, that logic is flawed.
        # unique(b) = mask(b) \ (Union of others)
        # Union of others = S \ mask(b) is only true if mask(b) is the only one covering some parts? No.
        # Union of others is NOT simply S \ mask(b).
        # However, if we maintain S = OR of all active masks, then:
        # A t-subset is uniquely covered by b if:
        # 1. It is in mask(b)
        # 2. It is NOT in any other mask.
        # This is hard to compute directly from S alone without counting.
        
        # Let's stick to the counting approach or the O(N^2) mask intersection which might be okay for ~150 blocks.
        # N_blocks ~ 150. N^2 = 22500 operations. Each is a big-int AND/OR. Fast enough.
        
        def get_unique_masks(active_set):
            unique_masks = {}
            for idx in active_set:
                other_mask = 0
                for idx2 in active_set:
                    if idx2 != idx:
                        other_mask |= ksub_masks[idx2]
                unique_masks[idx] = ksub_masks[idx] & ~other_mask
            return unique_masks

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Calculate unique masks for all active blocks
            unique_masks = get_unique_masks(active_indices)
            
            # Find redundant blocks (unique mask is 0)
            to_remove = [idx for idx, um in unique_masks.items() if um == 0]
            if to_remove:
                for idx in to_remove:
                    active_indices.remove(idx)
                improved = True
                continue
            
            # Try to find a swap that enables a removal
            # Sort by unique coverage size to prioritize "weak" blocks
            sorted_indices = sorted(active_indices, key=lambda x: unique_masks[x].bit_count())
            
            for target_idx in sorted_indices:
                if time.time() > deadline - 2.0:
                    break
                
                target_unique = unique_masks[target_idx]
                if target_unique == 0:
                    continue
                
                # Find a replacement block that covers target_unique
                replacement_idx = -1
                # Iterate through all blocks not in active set
                for j in range(n_ksubs):
                    if j in active_indices:
                        continue
                    if (ksub_masks[j] & target_unique) == target_unique:
                        replacement_idx = j
                        break
                
                if replacement_idx != -1:
                    # Perform swap
                    active_indices.remove(target_idx)
                    active_indices.add(replacement_idx)
                    
                    # Check if this swap allows any removal
                    new_unique_masks = get_unique_masks(active_indices)
                    
                    new_redundants = [idx for idx, um in new_unique_masks.items() if um == 0]
                    if new_redundants:
                        for idx in new_redundants:
                            active_indices.remove(idx)
                        improved = True
                        break # Restart the main loop with the new set
                    else:
                        # Revert if no immediate improvement
                        active_indices.remove(replacement_idx)
                        active_indices.add(target_idx)

        if active_indices:
            final_blocks = [list(all_ksubs[idx]) for idx in active_indices]
            final_mask = 0
            for idx in active_indices:
                final_mask |= ksub_masks[idx]
            if final_mask == (1 << n_tsubs) - 1:
                return final_blocks, list(active_indices)
        
        return initial_blocks, initial_indices

    # 4. Multi-start strategy
    best_blocks = None
    best_count = float('inf')
    
    # First run: Standard Greedy
    blocks, indices = greedy_init(randomize=False)
    polished_blocks, polished_indices = local_search(blocks, indices)
    if polished_blocks and len(polished_blocks) < best_count:
        best_blocks = polished_blocks
        best_count = len(polished_blocks)
        
    # Subsequent runs: Randomized Greedy
    # Limit number of starts based on time
    max_starts = 10
    for i in range(max_starts):
        if time.time() > deadline - 20.0:
            break
            
        # Randomized greedy
        rand_blocks, rand_indices = greedy_init(randomize=True)
        
        # Quick polish
        polished_rand_blocks, _ = local_search(rand_blocks, rand_indices)
        
        if polished_rand_blocks and len(polished_rand_blocks) < best_count:
            best_blocks = polished_rand_blocks
            best_count = len(polished_rand_blocks)

    if best_blocks is None:
        # Fallback to first result if multi-start failed or timed out
        blocks, indices = greedy_init(randomize=False)
        best_blocks, _ = local_search(blocks, indices)

    return best_blocks
