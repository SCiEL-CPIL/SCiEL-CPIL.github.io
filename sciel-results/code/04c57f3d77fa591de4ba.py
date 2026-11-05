"""Combining bitmask greedy with iterative local search and ILS perturbation."""
import itertools
import time
import random

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # Generate all k-blocks and precompute their coverage as bitmasks
    all_kblocks = list(itertools.combinations(points, k))
    block_masks = []
    for b in all_kblocks:
        mask = 0
        for ts in itertools.combinations(b, t):
            mask |= (1 << tsub_to_idx[ts])
        block_masks.append(mask)
    
    # Also store the list of t-subset indices for each block (for faster iteration)
    block_tsub_indices = []
    for mask in block_masks:
        indices = []
        m = mask
        while m:
            lsb = m & -m
            indices.append(lsb.bit_length() - 1)
            m ^= lsb
        block_tsub_indices.append(indices)
    
    def greedy_cover(uncovered_mask, max_blocks=None):
        """Greedy set cover using bitmask operations."""
        selected = []
        cur_mask = uncovered_mask
        while cur_mask != 0:
            if time.monotonic() > deadline - 5:
                break
            best_gain = 0
            best_idx = -1
            for i, mask in enumerate(block_masks):
                gain = (mask & cur_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == cur_mask.bit_count():
                        break
            if best_gain == 0:
                break
            selected.append(best_idx)
            cur_mask &= ~block_masks[best_idx]
            if max_blocks and len(selected) >= max_blocks:
                break
        return selected
    
    def remove_redundant(selected_indices):
        """Remove blocks that don't uniquely cover any t-subset."""
        if not selected_indices:
            return []
        
        # Count how many selected blocks cover each t-subset
        cover_count = [0] * num_tsubs
        for idx in selected_indices:
            for ts_idx in block_tsub_indices[idx]:
                cover_count[ts_idx] += 1
        
        improved = True
        while improved:
            improved = False
            new_selected = []
            for idx in selected_indices:
                # Check if this block is the only cover for any t-subset
                critical = False
                for ts_idx in block_tsub_indices[idx]:
                    if cover_count[ts_idx] == 1:
                        critical = True
                        break
                if critical:
                    new_selected.append(idx)
                else:
                    # Remove this block and update counts
                    for ts_idx in block_tsub_indices[idx]:
                        cover_count[ts_idx] -= 1
                    improved = True
            selected_indices = new_selected
        return selected_indices
    
    def polish(selected_indices):
        """Try to swap blocks to reduce total count while maintaining coverage."""
        if len(selected_indices) < 2:
            return selected_indices
        
        # Try removing one block and re-adding greedily
        for i in range(len(selected_indices)):
            if time.monotonic() > deadline - 3:
                break
            without_i = [j for j in selected_indices if j != i]
            # What's still uncovered?
            covered_mask = 0
            for j in without_i:
                covered_mask |= block_masks[j]
            uncovered = (1 << num_tsubs) - 1
            uncovered &= ~covered_mask
            if uncovered == 0:
                # We can remove block i entirely
                return remove_redundant(without_i)
            # Try to cover the gap with one block
            best_gain = 0
            best_add = -1
            for j, mask in enumerate(block_masks):
                if j in without_i:
                    continue
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_add = j
            if best_add != -1 and best_gain == uncovered.bit_count():
                # Perfect replacement
                return remove_redundant(without_i + [best_add])
        
        return selected_indices
    
    # Phase 1: Initial greedy cover
    full_mask = (1 << num_tsubs) - 1
    initial_selected = greedy_cover(full_mask)
    initial_selected = remove_redundant(initial_selected)
    initial_selected = polish(initial_selected)
    
    best_selected = initial_selected[:]
    best_count = len(best_selected)
    
    # Phase 2: ILS - Iterated Local Search
    if time.monotonic() < deadline - 10:
        attempts = 0
        while time.monotonic() < deadline - 5 and attempts < 50:
            attempts += 1
            # Perturb: randomly remove some blocks
            perturbed = best_selected[:]
            n_remove = max(1, len(perturbed) // 10)
            for _ in range(n_remove):
                if perturbed:
                    idx_to_remove = random.randrange(len(perturbed))
                    perturbed.pop(idx_to_remove)
            
            # Re-add greedily to restore coverage
            covered_mask = 0
            for idx in perturbed:
                covered_mask |= block_masks[idx]
            uncovered = full_mask & ~covered_mask
            
            if uncovered != 0:
                # Greedily add blocks to cover uncovered
                add_selected = greedy_cover(uncovered)
                perturbed.extend(add_selected)
            
            # Polish
            perturbed = remove_redundant(perturbed)
            perturbed = polish(perturbed)
            
            if len(perturbed) < best_count:
                best_selected = perturbed[:]
                best_count = len(best_selected)
    
    # Convert indices to blocks
    blocks = [list(all_kblocks[i]) for i in best_selected]
    return blocks
