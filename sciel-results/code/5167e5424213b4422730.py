# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 148.0
    rng = random.Random(42)

    # Precompute all t-subsets as bitmasks
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    n_tsub = len(tsub_masks)
    idx_of = {m: i for i, m in enumerate(tsub_masks)}

    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)

    # Precompute t-subset indices for each k-subset
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        indices = []
        for sub in itertools.combinations(comb, t):
            m = 0
            for p in sub:
                m |= (1 << p)
            indices.append(idx_of[m])
        k_sub_tsub_indices.append(indices)

    # Build mask-to-index lookup for k-subsets
    mask_to_idx = {}
    for i, comb in enumerate(all_k_subsets):
        m = 0
        for p in comb:
            m |= (1 << p)
        mask_to_idx[m] = i

    # Flat array of t-subset indices with offsets
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)

    # Build inverse mapping: for each t-subset, which k-subsets cover it
    # tsub_to_ksubs[tsub_idx] = list of k-subset indices that contain this t-subset
    tsub_to_ksubs = [[] for _ in range(n_tsub)]
    for i in range(n_ksub):
        start = offsets_arr[i]
        end = offsets_arr[i + 1]
        for idx in all_indices_arr[start:end]:
            tsub_to_ksubs[int(idx)].append(i)

    def greedy_cover(uncovered, deadline):
        """Greedy set cover with efficient gain computation."""
        blocks = []
        selected_set = set()
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        
        # Track which blocks are still available
        available = np.ones(n_ksub, dtype=bool)
        
        while uncovered.any() and time.time() < deadline:
            # Compute gains for all available blocks
            gains = np.zeros(n_ksub, dtype=np.int32)
            for i in range(n_ksub):
                if not available[i]:
                    continue
                start = offsets_arr[i]
                end = offsets_arr[i + 1]
                if start < end:
                    gains[i] = np.sum(uncovered[all_indices_arr[start:end]])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where((gains == max_gain) & available)[0]
            best_idx = candidates[0]
            blocks.append(best_idx)
            selected_set.add(best_idx)
            available[best_idx] = False
            
            start = offsets_arr[best_idx]
            end = offsets_arr[best_idx + 1]
            idxs = all_indices_arr[start:end]
            cover_count[idxs] += 1
            uncovered[idxs] = False
        
        return blocks, selected_set, cover_count

    def try_redundant_removal(blocks, selected_set, cover_count, deadline):
        """Remove any redundant blocks (those with no t-subset uniquely covered)."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                start = offsets_arr[b_idx]
                end = offsets_arr[b_idx + 1]
                idxs = all_indices_arr[start:end]
                if np.any(cover_count[idxs] == 1):
                    continue  # Not redundant
                # Remove
                cover_count[idxs] -= 1
                blocks.pop(i)
                selected_set.remove(b_idx)
                changed = True
                break
        return blocks, selected_set, cover_count

    def try_single_swap(blocks, selected_set, cover_count, deadline):
        """Try to replace one block with another that covers more uncovered t-subsets."""
        improved = False
        n_trials = 200
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Pick a random block to potentially replace
            bi = rng.randint(0, len(blocks) - 1)
            old_idx = blocks[bi]
            
            # Find t-subsets that would become uncovered if we remove this block
            start_old = offsets_arr[old_idx]
            end_old = offsets_arr[old_idx + 1]
            idxs_old = all_indices_arr[start_old:end_old]
            unique_tsubs = [int(idx) for idx in idxs_old if cover_count[idx] == 1]
            
            if not unique_tsubs:
                # Block is redundant, remove it
                cover_count[idxs_old] -= 1
                blocks.pop(bi)
                selected_set.remove(old_idx)
                improved = True
                continue
            
            # Find candidate replacement blocks that cover all unique_tsubs
            # and are not already selected
            candidates = []
            for tsub_idx in unique_tsubs:
                for cand in tsub_to_ksubs[tsub_idx]:
                    if cand not in selected_set and cand != old_idx:
                        candidates.append(cand)
            
            if not candidates:
                continue
            
            # Try each candidate
            for cand in candidates[:50]:  # Limit to 50 candidates for speed
                if time.time() > deadline:
                    break
                
                # Check if this candidate covers all unique_tsubs
                start_cand = offsets_arr[cand]
                end_cand = offsets_arr[cand + 1]
                idxs_cand = set(int(idx) for idx in all_indices_arr[start_cand:end_cand])
                
                if all(t in idxs_cand for t in unique_tsubs):
                    # Check if this swap is beneficial (covers more total t-subsets)
                    old_coverage = len(set(int(idx) for idx in all_indices_arr[start_old:end_old]))
                    cand_coverage = len(idxs_cand)
                    
                    if cand_coverage >= old_coverage:
                        # Perform the swap
                        cover_count[idxs_old] -= 1
                        cover_count[all_indices_arr[start_cand:end_cand]] += 1
                        blocks[bi] = cand
                        selected_set.remove(old_idx)
                        selected_set.add(cand)
                        improved = True
                        break
        
        return improved

    def try_k_for_k1_exchange(blocks, selected_set, cover_count, deadline):
        """
        Try to remove k blocks and add k-1 new blocks (net reduction of 1).
        """
        improved = False
        for k_remove in [2, 3]:
            if time.time() > deadline:
                break
            if len(blocks) < k_remove:
                break
            
            n_trials = 30
            for trial in range(n_trials):
                if time.time() > deadline:
                    break
                
                # Randomly select k_remove blocks to remove
                to_remove = rng.sample(range(len(blocks)), k_remove)
                
                # Find all t-subsets that become uncovered if we remove these blocks
                affected_tsubs = set()
                for bi in to_remove:
                    b_idx = blocks[bi]
                    start = offsets_arr[b_idx]
                    end = offsets_arr[b_idx + 1]
                    idxs = all_indices_arr[start:end]
                    for idx in idxs:
                        if cover_count[idx] == 1:
                            affected_tsubs.add(int(idx))
                
                if not affected_tsubs:
                    # All removed blocks are redundant
                    for bi in sorted(to_remove, reverse=True):
                        b_idx = blocks[bi]
                        start = offsets_arr[b_idx]
                        end = offsets_arr[b_idx + 1]
                        idxs = all_indices_arr[start:end]
                        cover_count[idxs] -= 1
                        blocks.pop(bi)
                        selected_set.remove(b_idx)
                    improved = True
                    break
                
                # Try to find k_remove - 1 blocks that cover all affected t-subsets
                n_add = k_remove - 1
                
                # Build candidate list from t-subset to k-subset mapping
                candidate_set = set()
                for tsub_idx in affected_tsubs:
                    for cand in tsub_to_ksubs[tsub_idx]:
                        if cand not in selected_set:
                            candidate_set.add(cand)
                
                if len(candidate_set) < n_add:
                    continue
                
                candidates = list(candidate_set)
                
                # Try to find n_add candidates that cover all affected_tsubs
                found_cover = False
                n_add_trials = min(100, max(10, len(candidates) // 2))
                for add_trial in range(n_add_trials):
                    if time.time() > deadline:
                        break
                    
                    add_blocks = rng.sample(candidates, min(n_add, len(candidates)))
                    
                    if len(add_blocks) < n_add:
                        continue
                    
                    # Verify: do these add_blocks cover all affected t-subsets?
                    covered_by_add = set()
                    for ab in add_blocks:
                        start = offsets_arr[ab]
                        end = offsets_arr[ab + 1]
                        idxs = all_indices_arr[start:end]
                        for idx in idxs:
                            covered_by_add.add(int(idx))
                    
                    if affected_tsubs.issubset(covered_by_add):
                        # Valid exchange! Apply it.
                        for bi in sorted(to_remove, reverse=True):
                            b_idx = blocks[bi]
                            start = offsets_arr[b_idx]
                            end = offsets_arr[b_idx + 1]
                            idxs = all_indices_arr[start:end]
                            cover_count[idxs] -= 1
                            blocks.pop(bi)
                            selected_set.remove(b_idx)
                        
                        for ab in add_blocks:
                            blocks.append(ab)
                            selected_set.add(ab)
                            start = offsets_arr[ab]
                            end = offsets_arr[ab + 1]
                            idxs = all_indices_arr[start:end]
                            cover_count[idxs] += 1
                        
                        found_cover = True
                        improved = True
                        break
                
                if found_cover:
                    break
        
        return improved

    # Phase 1: Greedy initialization
    uncovered = np.ones(n_tsub, dtype=bool)
    blocks, selected_set, cover_count = greedy_cover(uncovered, deadline - 40.0)
    
    # Phase 2: Remove redundant blocks
    blocks, selected_set, cover_count = try_redundant_removal(blocks, selected_set, cover_count, deadline - 30.0)
    
    # Phase 3: Local search
    best_blocks = list(blocks)
    best_count = len(blocks)
    best_selected = set(selected_set)
    best_cover = cover_count.copy()
    
    no_improve = 0
    while time.time() < deadline - 5.0:
        # Try redundant removal
        blocks, selected_set, cover_count = try_redundant_removal(blocks, selected_set, cover_count, deadline - 5.0)
        
        # Try single swaps
        improved = try_single_swap(blocks, selected_set, cover_count, deadline - 5.0)
        
        # Try k-for-(k-1) exchanges
        if time.time() < deadline - 5.0:
            improved2 = try_k_for_k1_exchange(blocks, selected_set, cover_count, deadline - 5.0)
            improved = improved or improved2
        
        if len(blocks) < best_count:
            best_blocks = list(blocks)
            best_count = len(blocks)
            best_selected = set(selected_set)
            best_cover = cover_count.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 30:
                break
    
    # Use best found
    blocks = best_blocks
    selected_set = best_selected
    cover_count = best_cover
    
    # Final redundant removal
    blocks, selected_set, cover_count = try_redundant_removal(blocks, selected_set, cover_count, deadline - 2.0)
    
    # Final single swap pass
    if time.time() < deadline - 1.5:
        try_single_swap(blocks, selected_set, cover_count, deadline - 1.0)
    
    # Verify and repair if needed
    uncovered = np.ones(n_tsub, dtype=bool)
    for b_idx in blocks:
        start = offsets_arr[b_idx]
        end = offsets_arr[b_idx + 1]
        idxs = all_indices_arr[start:end]
        uncovered[idxs] = False
    
    uncovered_indices = np.where(uncovered)[0]
    for idx in uncovered_indices:
        if time.time() > deadline - 0.5:
            break
        tsub_mask = tsub_masks[idx]
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        if need > 0:
            extra = rng.sample(remaining_points, need)
        else:
            extra = []
        block = have + extra
        block.sort()
        mask = 0
        for p in block:
            mask |= (1 << p)
        b_idx = mask_to_idx.get(mask)
        if b_idx is not None and b_idx not in selected_set:
            blocks.append(b_idx)
            selected_set.add(b_idx)
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx + 1]
            idxs = all_indices_arr[start:end]
            cover_count[idxs] += 1
            uncovered[idxs] = False
    
    # Final result
    result = []
    for idx in blocks:
        result.append(list(all_k_subsets[idx]))
    
    return result

# EVOLVE-BLOCK-END
