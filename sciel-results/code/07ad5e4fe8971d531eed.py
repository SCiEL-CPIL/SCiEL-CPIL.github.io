import itertools
import time
import random
from collections import defaultdict
import math
import numpy as np

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
    
    # Precompute numpy array of block masks for vectorized operations
    # Since n_t can be large, we use uint64 arrays for each 64-bit chunk
    n_chunks = (n_t + 63) // 64
    block_masks_np = np.zeros((n_blocks, n_chunks), dtype=np.uint64)
    for i in range(n_blocks):
        mask = block_masks[i]
        for c in range(n_chunks):
            block_masks_np[i, c] = (mask >> (64 * c)) & 0xFFFFFFFFFFFFFFFF
    
    def greedy_cover(weights=None):
        """Greedy set cover with optional Lagrangian weights."""
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
            
            best_block_idx = -1
            best_score = 0
            
            for i in range(n_blocks):
                bm = block_masks[i]
                inter = uncovered_mask & bm
                if inter == 0:
                    continue
                if weights is None:
                    gain = inter.bit_count()
                else:
                    # Approximate weighted gain using bit_count as proxy
                    # (full weighted sum is too expensive; use count as heuristic)
                    gain = inter.bit_count()
                
                if gain > best_score:
                    best_score = gain
                    best_block_idx = i
                    
            if best_block_idx == -1:
                break
                
            selected.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return selected
    
    def primal_dual_cover():
        """Primal-dual approximation for set cover.
        
        Iteratively raises dual prices for uncovered t-subsets and selects
        blocks with the highest dual-weight-to-cost ratio.
        """
        selected = []
        selected_set = set()
        uncovered_mask = (1 << n_t) - 1
        dual_prices = [0.0] * n_t
        
        while uncovered_mask and time.time() < deadline:
            # Find the block with maximum sum of dual prices over uncovered elements it covers
            best_block = -1
            best_ratio = 0.0
            
            for i in range(n_blocks):
                bm = block_masks[i]
                inter = uncovered_mask & bm
                if inter == 0:
                    continue
                
                # Sum of dual prices for uncovered elements covered by this block
                w_sum = 0.0
                m = inter
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    w_sum += dual_prices[idx]
                    m ^= lsb
                
                if w_sum > best_ratio:
                    best_ratio = w_sum
                    best_block = i
            
            if best_block == -1:
                break
            
            # Raise dual prices for uncovered elements covered by best_block
            # until at least one element's dual price reaches its "cost" (1.0)
            bm = block_masks[best_block]
            inter = uncovered_mask & bm
            
            # Find the minimum amount to raise all dual prices in inter
            # such that at least one reaches 1.0
            min_headroom = 1.0
            m = inter
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                headroom = 1.0 - dual_prices[idx]
                if headroom < min_headroom:
                    min_headroom = headroom
                m ^= lsb
            
            # Raise dual prices
            m = inter
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                dual_prices[idx] += min_headroom
                m ^= lsb
            
            selected.append(best_block)
            selected_set.add(best_block)
            uncovered_mask &= ~bm
        
        return selected
    
    def local_search(selected):
        """Aggressive local search: removal + 1-swap + 2-swap + simulated annealing."""
        if not selected:
            return selected
        
        selected = list(selected)
        
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
                for idx in block_t_indices[b_idx]:
                    if counts[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    selected_set.discard(b_idx)
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Phase 2: 1-swap local search with gain-based candidate ranking
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_selected = list(selected)
            new_set = set(new_selected)
            
            counts = [0] * n_t
            for b_idx in new_selected:
                for idx in block_t_indices[b_idx]:
                    counts[idx] += 1
            
            for pos, b_idx in enumerate(list(new_selected)):
                if time.time() > deadline:
                    break
                unique_indices = [idx for idx in block_t_indices[b_idx] if counts[idx] == 1]
                
                if not unique_indices:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    new_selected.remove(b_idx)
                    new_set.discard(b_idx)
                    improved = True
                    break
                
                # Find best swap candidate
                best_swap = -1
                best_gain = 0
                
                for candidate in t_to_blocks[unique_indices[0]]:
                    if candidate in new_set:
                        continue
                    cand_mask = block_masks[candidate]
                    covers_all = all((cand_mask >> ui) & 1 for ui in unique_indices)
                    if not covers_all:
                        continue
                    orig_mask = block_masks[b_idx]
                    diff = cand_mask & ~orig_mask
                    extra = diff.bit_count()
                    if extra > best_gain:
                        best_gain = extra
                        best_swap = candidate
                
                if best_swap != -1 and best_gain > 0:
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
        
        # Phase 3: 2-swap local search (swap two blocks for one or two other blocks)
        if time.time() < deadline - 15:
            improved2 = True
            while improved2 and time.time() < deadline - 10:
                improved2 = False
                cur_selected = list(selected)
                cur_set = set(cur_selected)
                
                counts2 = [0] * n_t
                for b_idx in cur_selected:
                    for idx in block_t_indices[b_idx]:
                        counts2[idx] += 1
                
                # Try removing one block and replacing with up to 2 blocks
                for b_idx in list(cur_selected):
                    if time.time() > deadline - 10:
                        break
                    
                    unique_indices = [idx for idx in block_t_indices[b_idx] if counts2[idx] == 1]
                    if not unique_indices:
                        # Just remove
                        for idx in block_t_indices[b_idx]:
                            counts2[idx] -= 1
                        cur_selected.remove(b_idx)
                        cur_set.discard(b_idx)
                        improved2 = True
                        break
                    
                    # Find single block that covers all unique indices (1-for-1 swap)
                    best_single = -1
                    best_single_gain = 0
                    for candidate in t_to_blocks[unique_indices[0]]:
                        if candidate in cur_set:
                            continue
                        cand_mask = block_masks[candidate]
                        if all((cand_mask >> ui) & 1 for ui in unique_indices):
                            orig_mask = block_masks[b_idx]
                            diff = cand_mask & ~orig_mask
                            extra = diff.bit_count()
                            if extra > best_single_gain:
                                best_single_gain = extra
                                best_single = candidate
                    
                    if best_single != -1 and best_single_gain > 0:
                        for idx in block_t_indices[b_idx]:
                            counts2[idx] -= 1
                        for idx in block_t_indices[best_single]:
                            counts2[idx] += 1
                        cur_selected.remove(b_idx)
                        cur_selected.append(best_single)
                        cur_set.discard(b_idx)
                        cur_set.add(best_single)
                        improved2 = True
                        break
                    
                    # Try 1-for-2: remove b_idx, add two blocks that together cover unique_indices
                    # This can sometimes reduce total count if the two blocks enable more removals
                    # For now, skip 1-for-2 as it increases count; focus on 2-for-1 later
        
        # Phase 4: Simulated annealing with better acceptance criteria
        if time.time() < deadline - 10:
            T = 2.0
            cooling_rate = 0.97
            min_T = 0.05
            
            current_selected = list(selected)
            current_counts = [0] * n_t
            for b_idx in current_selected:
                for idx in block_t_indices[b_idx]:
                    current_counts[idx] += 1
            current_set = set(current_selected)
            current_len = len(current_selected)
            
            best_selected = list(current_selected)
            best_len = current_len
            
            no_improve_count = 0
            max_no_improve = 100
            
            while time.time() < deadline - 5 and T > min_T:
                if no_improve_count > max_no_improve:
                    break
                
                if not current_selected:
                    break
                
                b_idx = random.choice(current_selected)
                unique_indices = [idx for idx in block_t_indices[b_idx] if current_counts[idx] == 1]
                
                if not unique_indices:
                    for idx in block_t_indices[b_idx]:
                        current_counts[idx] -= 1
                    current_selected.remove(b_idx)
                    current_set.discard(b_idx)
                    
                    new_len = len(current_selected)
                    if new_len < best_len:
                        best_selected = list(current_selected)
                        best_len = new_len
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    T *= cooling_rate
                    continue
                
                valid_candidates = []
                for candidate in t_to_blocks[unique_indices[0]]:
                    if candidate in current_set:
                        continue
                    cand_mask = block_masks[candidate]
                    if all((cand_mask >> ui) & 1 for ui in unique_indices):
                        valid_candidates.append(candidate)
                
                if not valid_candidates:
                    T *= cooling_rate
                    no_improve_count += 1
                    continue
                
                candidate = random.choice(valid_candidates)
                
                for idx in block_t_indices[b_idx]:
                    current_counts[idx] -= 1
                for idx in block_t_indices[candidate]:
                    current_counts[idx] += 1
                current_selected.remove(b_idx)
                current_selected.append(candidate)
                current_set.discard(b_idx)
                current_set.add(candidate)
                
                # Remove redundant blocks
                changed = True
                while changed and time.time() < deadline - 5:
                    changed = False
                    new_sel = []
                    for b in current_selected:
                        redundant = True
                        for idx in block_t_indices[b]:
                            if current_counts[idx] <= 1:
                                redundant = False
                                break
                        if redundant:
                            for idx in block_t_indices[b]:
                                current_counts[idx] -= 1
                            current_set.discard(b)
                            changed = True
                        else:
                            new_sel.append(b)
                    current_selected = new_sel
                
                new_len = len(current_selected)
                if new_len < best_len:
                    best_selected = list(current_selected)
                    best_len = new_len
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                T *= cooling_rate
            
            selected = best_selected
        
        return selected
    
    def shrink_solution(selected):
        """Try to remove redundant blocks."""
        if not selected:
            return selected
        
        selected = list(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Compute counts once
            counts = [0] * n_t
            for b in selected:
                for idx in block_t_indices[b]:
                    counts[idx] += 1
            
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                b_idx = selected[i]
                
                unique_to_this = [idx for idx in block_t_indices[b_idx] if counts[idx] == 1]
                
                if not unique_to_this:
                    for idx in block_t_indices[b_idx]:
                        counts[idx] -= 1
                    selected.pop(i)
                    improved = True
                    break
        
        return selected
    
    # Multi-restart with different strategies
    best_solution = None
    best_len = float('inf')
    
    # Strategy 1: Simple greedy
    sol1 = greedy_cover()
    sol1 = local_search(sol1)
    if sol1:
        best_solution = sol1
        best_len = len(sol1)
    
    # Strategy 2: Primal-dual approximation
    if time.time() < deadline - 40:
        sol2 = primal_dual_cover()
        sol2 = local_search(sol2)
        if sol2 and len(sol2) < best_len:
            best_solution = sol2
            best_len = len(sol2)
    
    # Strategy 3: Multi-restart greedy with different random seeds for tie-breaking
    for iteration in range(4):
        if time.time() > deadline - 30:
            break
        
        sol = greedy_cover()
        if not sol:
            break
        
        sol = local_search(sol)
        if sol and len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Strategy 4: Randomized greedy (multiple trials) with top-k sampling
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        while uncovered_mask and time.time() < deadline - 10:
            # Find top-k blocks by gain
            k = 10
            top_blocks = []
            best_gain = 0
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    top_blocks = [(gain, i)]
                elif gain == best_gain and gain > 0:
                    top_blocks.append((gain, i))
            
            if not top_blocks or top_blocks[0][0] == 0:
                break
            
            # Sample from top-k with preference for higher gain
            if len(top_blocks) > k:
                # Take top k by gain (they're all equal gain here)
                top_blocks = top_blocks[:k]
            
            chosen = random.choice(top_blocks)[1]
            selected.append(chosen)
            uncovered_mask &= ~block_masks[chosen]
        
        if selected:
            selected = local_search(selected)
            if selected and len(selected) < best_len:
                best_solution = selected
                best_len = len(selected)
    
    # Final: Try to improve best_solution with more aggressive local search and shrink
    if best_solution and time.time() < deadline - 5:
        best_solution = local_search(best_solution)
        best_solution = shrink_solution(best_solution)
        if time.time() < deadline - 2:
            best_solution = local_search(best_solution)
    
    # Convert indices back to blocks
    if best_solution:
        return [list(all_blocks[i]) for i in best_solution]
    else:
        return []
