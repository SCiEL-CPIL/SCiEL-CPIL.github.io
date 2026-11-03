import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    # Also store list of t-subset indices for each block (for faster count updates)
    block_t_indices = [[] for _ in range(n_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        indices = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices
    
    # Precompute for each t-subset, which blocks cover it (reverse index)
    # t_to_blocks[j] = list of block indices that cover t_subset j
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_blocks):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].append(i)
    
    def greedy_cover(weights=None):
        """Greedy set cover with optional Lagrangian weights.
        If weights is None, uses simple max-gain.
        If weights is a list, uses weighted gain: sum of weights for uncovered t-subsets covered.
        """
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            best_block_idx = -1
            best_score = 0
            
            for i in range(n_blocks):
                bm = block_masks[i]
                if bm == 0:
                    continue
                # Compute gain
                if weights is None:
                    gain = (uncovered_mask & bm).bit_count()
                else:
                    # Weighted gain: sum of weights for uncovered t-subsets in this block
                    # Efficient: iterate over set bits of (uncovered_mask & bm)
                    inter = uncovered_mask & bm
                    if inter == 0:
                        continue
                    # For small numbers of bits, iterate
                    w_sum = 0
                    m = inter
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        w_sum += weights[idx]
                        m ^= lsb
                    gain = w_sum
                
                if gain > best_score:
                    best_score = gain
                    best_block_idx = i
                    
            if best_block_idx == -1:
                break
                
            selected.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return selected
    
    def local_search(selected):
        """Aggressive local search: removal + swap-based improvement."""
        if not selected:
            return selected
        
        # Initialize counts
        counts = [0] * n_t
        for b_idx in selected:
            for idx in block_t_indices[b_idx]:
                counts[idx] += 1
        
        selected_set = set(selected)
        
        # Phase 1: Iterative removal of redundant blocks
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for b_idx in selected:
                mask = block_masks[b_idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    selected_set.discard(b_idx)
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Phase 2: Swap-based local search
        # Try to replace one block with another that covers all its unique t-subsets
        # and also covers some additional uncovered t-subsets, or simply reduces count
        # More precisely: try replacing block A with block B if:
        #   - Every t-subset covered only by A is also covered by B
        #   - This doesn't increase the number of blocks (it's 1-for-1, so count stays same)
        #   - But we want to enable further removals. So we just do swaps that are "neutral"
        #     or lead to more redundancy.
        
        # Better approach: try to remove block A and add block B if B covers all
        # t-subsets that A uniquely covers, and B is not already selected.
        # This is a 1-move swap.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            # Recompute counts for current selected
            counts = [0] * n_t
            for b_idx in new_selected:
                for idx in block_t_indices[b_idx]:
                    counts[idx] += 1
            
            # For each block in selected, check if we can swap it out
            for pos, b_idx in enumerate(list(new_selected)):
                if time.time() > deadline:
                    break
                # Find t-subsets uniquely covered by b_idx
                unique_indices = []
                m = block_masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        unique_indices.append(idx)
                    m ^= lsb
                
                if not unique_indices:
                    # This block is redundant, remove it
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    new_selected.remove(b_idx)
                    new_set.discard(b_idx)
                    improved = True
                    break
                
                # Try to find a block B (not in new_set) that covers all unique_indices
                # and is "good" (covers many total t-subsets)
                best_swap = -1
                best_gain = 0
                for candidate in t_to_blocks[unique_indices[0]]:
                    if candidate in new_set:
                        continue
                    # Check if candidate covers all unique_indices
                    covers_all = True
                    for ui in unique_indices:
                        if not (block_masks[candidate] >> ui) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Compute additional gain: t-subsets covered by candidate but not by b_idx
                    # that are currently uncovered or under-covered
                    extra = 0
                    cand_mask = block_masks[candidate]
                    orig_mask = block_masks[b_idx]
                    # t-subsets in candidate but not in original block
                    diff = cand_mask & ~orig_mask
                    # Count how many of these are currently covered by exactly 1 block
                    # (so adding candidate would increase their count, which is good for future removals)
                    m = diff
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if counts[idx] == 0:
                            # This would be a new coverage, very good
                            extra += 10  # high priority
                        elif counts[idx] == 1:
                            extra += 1
                        m ^= lsb
                    
                    if extra > best_gain:
                        best_gain = extra
                        best_swap = candidate
                
                if best_swap != -1 and best_gain > 0:
                    # Perform the swap
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    for idx in block_t_indices[best_swap]:
                        counts[idx] += 1
                    new_selected.remove(b_idx)
                    new_selected.append(best_swap)
                    new_set.discard(b_idx)
                    new_set.add(best_swap)
                    improved = True
                    break
        
            selected = new_selected
        
        return selected
    
    # Multi-restart with Lagrangian relaxation
    best_solution = None
    best_len = float('inf')
    
    # First: simple greedy
    sol1 = greedy_cover()
    sol1 = local_search(sol1)
    if sol1:
        best_solution = sol1
        best_len = len(sol1)
    
    # Second: Lagrangian-guided greedy
    # Initialize weights: high for uncovered t-subsets
    # Iteratively: run greedy with weights, update weights based on uncovered
    weights = [1.0] * n_t
    for iteration in range(5):
        if time.time() > deadline - 30:
            break
        
        # Run greedy with current weights
        sol = greedy_cover(weights)
        if not sol:
            break
        
        # Check which t-subsets are uncovered
        covered_mask = 0
        for b_idx in sol:
            covered_mask |= block_masks[b_idx]
        
        # Update weights: increase for uncovered, decrease for well-covered
        new_weights = [0.0] * n_t
        m = (1 << n_t) - 1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if not (covered_mask >> idx) & 1:
                new_weights[idx] = weights[idx] * 2.0
            else:
                new_weights[idx] = weights[idx] * 0.5
            m ^= lsb
        weights = new_weights
        
        sol = local_search(sol)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Third: Randomized greedy (multiple trials)
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        
        # Randomized greedy: among top-k blocks by gain, pick randomly
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        while uncovered_mask and time.time() < deadline - 10:
            # Compute gains for all blocks
            gains = []
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                gains.append(gain)
            
            if max(gains) == 0:
                break
            
            # Pick from top 5% of blocks
            threshold = max(gains) * 0.8
            candidates = [i for i in range(n_blocks) if gains[i] >= threshold]
            if not candidates:
                candidates = [i for i in range(n_blocks) if gains[i] > 0]
            
            if not candidates:
                break
            
            chosen = random.choice(candidates)
            selected.append(chosen)
            uncovered_mask &= ~block_masks[chosen]
        
        if selected:
            selected = local_search(selected)
            if selected and len(selected) < best_len:
                best_solution = selected
                best_len = len(selected)
    
    # Final: Try to improve best_solution with more aggressive local search
    if best_solution and time.time() < deadline - 5:
        # Try more swap iterations
        best_solution = local_search(best_solution)
    
    # Convert indices back to blocks
    if best_solution:
        return [list(all_blocks[i]) for i in best_solution]
    else:
        # Fallback: return empty (shouldn't happen)
        return []
