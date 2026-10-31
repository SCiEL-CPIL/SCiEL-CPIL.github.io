"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Precompute all t-subsets and map to index
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # Precompute all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # Precompute coverage mask for each k-subset using Python int bitsets
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # Initialize uncovered bitset (all 1s)
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs

    blocks = []
    used_blocks = set()

    # --- Phase 1: Greedy set cover ---
    while remaining > 0:
        if time.time() > deadline - 20.0:
            break

        best_j = -1
        best_gain = 0

        # Scan all candidates to find max gain
        for j in range(n_ksubs):
            if j in used_blocks:
                continue
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain == 0:
            break

        block = list(all_ksubs[best_j])
        blocks.append(block)
        used_blocks.add(best_j)
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # --- Phase 2: Redundancy removal (local search) ---
    if time.time() < deadline - 5.0 and blocks:
        # Build coverage count for each t-subset
        # cover_count[i] = number of blocks in `blocks` that cover t-subset i
        cover_count = [0] * n_tsubs
        for b_idx in range(len(blocks)):
            mask = ksub_masks[blocks.index(tuple(blocks[b_idx]))] if tuple(blocks[b_idx]) in [tuple(b) for b in all_ksubs] else 0
            # Wait, blocks stores lists, all_ksubs stores tuples.
            # Let's store indices instead to be efficient.
            pass
        
        # Let's re-do Phase 1 storing indices for efficiency in Phase 2
        # Actually, let's just use the masks directly.
        # We need to map block list -> index in all_ksubs.
        # Since blocks are generated from all_ksubs, we can store the index.
        
        # Let's restart Phase 1 logic slightly differently to keep indices, or just map back.
        # Mapping back is O(n) per block, which is fine for ~150 blocks.
        
        # Better approach: keep a list of selected indices
        # Let's refactor the greedy loop to store indices.
        
        # Re-implementing Phase 1 with index tracking for Phase 2 benefit
        # But we already have `blocks`. Let's just compute cover_count from `blocks`.
        
        # To compute cover_count efficiently, we need the mask for each block in `blocks`.
        # We can find the index in all_ksubs.
        ksub_to_index = {ks: i for i, ks in enumerate(all_ksubs)}
        
        selected_indices = []
        for b in blocks:
            selected_indices.append(ksub_to_index[tuple(b)])
            
        cover_count = [0] * n_tsubs
        for idx in selected_indices:
            mask = ksub_masks[idx]
            # Iterate set bits in mask
            m = mask
            while m:
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb

        # Try to remove blocks that are redundant
        # A block is redundant if for all t-subs it covers, cover_count[t_sub] > 1
        # We iterate through blocks, and if redundant, remove it and update counts.
        
        # To make this effective, we should try removing blocks that cover the most "critical" (count==1) subsets?
        # No, we want to remove blocks that do NOT cover any critical subset.
        # i.e., if all t-subs covered by block B have count > 1, B is redundant.
        
        new_selected_indices = []
        changed = True
        while changed:
            changed = False
            new_selected_indices = []
            for idx in selected_indices:
                mask = ksub_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & (-m)
                    i = lsb.bit_length() - 1
                    if cover_count[i] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    # Remove this block
                    m = mask
                    while m:
                        lsb = m & (-m)
                        i = lsb.bit_length() - 1
                        cover_count[i] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_selected_indices.append(idx)
            
            selected_indices = new_selected_indices

        # Update blocks list
        blocks = [list(all_ksubs[idx]) for idx in selected_indices]

    # --- Phase 3: 1-for-1 Swap (Local Search) ---
    if time.time() < deadline - 5.0 and blocks:
        # Rebuild cover_count and selected_indices
        ksub_to_index = {ks: i for i, ks in enumerate(all_ksubs)}
        selected_indices = [ksub_to_index[tuple(b)] for b in blocks]
        
        cover_count = [0] * n_tsubs
        for idx in selected_indices:
            mask = ksub_masks[idx]
            m = mask
            while m:
                lsb = m & (-m)
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb

        # Try swaps: replace one selected block with one unselected block if it reduces size or maintains size with better properties?
        # Standard swap: Remove block A, Add block B.
        # This is valid if B covers all t-subs that A uniquely covered (count==1 after removing A).
        # And we want to do this if it allows further reductions or just to escape local minima.
        # Since we want to minimize count, a swap doesn't reduce count.
        # However, a swap might enable a subsequent redundancy removal.
        
        # Let's do a simple hill climbing:
        # 1. Try to remove a block (redundancy check).
        # 2. If no block can be removed, try a swap:
        #    For each selected block A:
        #      Calculate unique_tsubs_A = {t in A | count[t]==1}
        #      For each unselected block B:
        #        If B covers all unique_tsubs_A:
        #           Swap A -> B
        #           Update counts
        #           Try to remove redundancies again
        #           If size decreased, keep. Else revert.
        
        # This is expensive. Let's limit iterations.
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 2.0:
            improved = False
            iterations += 1
            
            # First pass: remove redundancies
            new_selected_indices = []
            for idx in selected_indices:
                mask = ksub_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & (-m)
                    i = lsb.bit_length() - 1
                    if cover_count[i] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                if is_redundant:
                    m = mask
                    while m:
                        lsb = m & (-m)
                        i = lsb.bit_length() - 1
                        cover_count[i] -= 1
                        m ^= lsb
                    improved = True
                else:
                    new_selected_indices.append(idx)
            selected_indices = new_selected_indices
            
            if not selected_indices:
                break

            # If no redundancy removed, try a swap
            if not improved:
                swapped = False
                # Pick a random selected block to try swapping out? Or iterate?
                # Iterating all pairs is O(N^2). N~150, N^2=22500.
                # For each pair, check coverage.
                
                # Let's pick a specific strategy:
                # Find a block A that has "many" unique subsets? No, we want to remove blocks with FEW unique subsets.
                # Actually, we want to find a swap that allows removal of another block.
                
                # Simplified Swap Logic:
                # For each i in selected_indices:
                #   unique_i = [t for t in mask_i if count[t]==1]
                #   If unique_i is empty, it's redundant (handled above).
                #   So unique_i is not empty.
                #   For each j not in selected_indices:
                #     If mask_j covers all unique_i:
                #        Perform swap i -> j
                #        Check if any block becomes redundant
                #        If yes, keep swap (size decreases)
                #        If no, revert swap
                
                # This is still O(N^2 * Cost).
                # Let's try a few random swaps to save time.
                
                for _ in range(50): # Try 50 random swaps
                    if time.time() > deadline - 2.0:
                        break
                    if not selected_indices:
                        break
                    
                    i_idx = random.randint(0, len(selected_indices)-1)
                    i = selected_indices[i_idx]
                    
                    # Find unique subsets for i
                    unique_subs = []
                    m = ksub_masks[i]
                    while m:
                        lsb = m & (-m)
                        t_idx = lsb.bit_length() - 1
                        if cover_count[t_idx] == 1:
                            unique_subs.append(t_idx)
                        m ^= lsb
                    
                    if not unique_subs:
                        continue # Should have been removed
                        
                    # Pick a random candidate j
                    j = random.randint(0, n_ksubs - 1)
                    if j in selected_indices:
                        continue
                        
                    # Check if j covers all unique_subs
                    mask_j = ksub_masks[j]
                    valid_swap = True
                    for t_idx in unique_subs:
                        if not (mask_j & (1 << t_idx)):
                            valid_swap = False
                            break
                            
                    if not valid_swap:
                        continue
                        
                    # Perform swap
                    # Remove i
                    m_i = ksub_masks[i]
                    while m_i:
                        lsb = m_i & (-m_i)
                        t_idx = lsb.bit_length() - 1
                        cover_count[t_idx] -= 1
                        m_i ^= lsb
                        
                    # Add j
                    m_j = mask_j
                    while m_j:
                        lsb = m_j & (-m_j)
                        t_idx = lsb.bit_length() - 1
                        cover_count[t_idx] += 1
                        m_j ^= lsb
                        
                    selected_indices[i_idx] = j
                    
                    # Check for new redundancies
                    new_selected = []
                    removed_count = 0
                    for idx in selected_indices:
                        mask = ksub_masks[idx]
                        is_redundant = True
                        m = mask
                        while m:
                            lsb = m & (-m)
                            t_idx = lsb.bit_length() - 1
                            if cover_count[t_idx] <= 1:
                                is_redundant = False
                                break
                            m ^= lsb
                        if is_redundant:
                            m = mask
                            while m:
                                lsb = m & (-m)
                                t_idx = lsb.bit_length() - 1
                                cover_count[t_idx] -= 1
                                m ^= lsb
                            removed_count += 1
                        else:
                            new_selected.append(idx)
                            
                    if removed_count > 0:
                        selected_indices = new_selected
                        improved = True
                        swapped = True
                        break # Restart loop
                    else:
                        # Revert swap
                        # Remove j
                        m_j = mask_j
                        while m_j:
                            lsb = m_j & (-m_j)
                            t_idx = lsb.bit_length() - 1
                            cover_count[t_idx] -= 1
                            m_j ^= lsb
                            
                        # Add i
                        m_i = ksub_masks[i]
                        while m_i:
                            lsb = m_i & (-m_i)
                            t_idx = lsb.bit_length() - 1
                            cover_count[t_idx] += 1
                            m_i ^= lsb
                            
                        selected_indices[i_idx] = i

        # Final update of blocks
        blocks = [list(all_ksubs[idx]) for idx in selected_indices]

    return blocks
# EVOLVE-BLOCK-END
