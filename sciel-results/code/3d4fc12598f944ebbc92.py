import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # 1. Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_masks[i] is a bitmask of t-subset indices covered by block i
    block_masks = [0] * n_k_blocks
    # Also store the list of t-subset indices for each block for fast iteration
    block_t_indices = [[] for _ in range(n_k_blocks)]
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices
        
    # Precompute for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(n_t_subs)]
    for i in range(n_k_blocks):
        for idx in block_t_indices[i]:
            t_to_blocks[idx].append(i)

    full_mask = (1 << n_t_subs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper: Fast Greedy construction using bitmask operations
    def greedy_construct(randomize=False, weight_mode='uniform'):
        """
        Greedy set cover with optional randomization.
        weight_mode: 'uniform', 'inverse_freq', 'inverse_sqrt'
        """
        if time.time() > deadline - 20:
            return []
        
        uncovered_mask = full_mask
        selected = []
        
        # Precompute weights if needed
        if weight_mode != 'uniform':
            # Count how many times each t-subset appears across all blocks
            # This is expensive, so only compute if needed
            pass
        
        while uncovered_mask != 0:
            if time.time() > deadline - 20:
                break
            
            best_idx = -1
            best_gain = -1
            best_candidates = []
            
            # For each block, compute gain = number of newly covered t-subsets
            # Use bitmask AND + bit_count for speed
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_idx == -1 or best_gain == 0:
                break
            
            if randomize and len(best_candidates) > 1:
                best_idx = random.choice(best_candidates)
            
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        
        return selected

    # Helper: Local Search with efficient 2-1 exchanges
    def local_search(selected):
        if not selected:
            return selected
        
        current = list(selected)
        
        # Build cover_count
        cover_count = [0] * n_t_subs
        for bi in current:
            for idx in block_t_indices[bi]:
                cover_count[idx] += 1
        
        # Set of selected blocks for fast lookup
        selected_set = set(current)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Phase 1: Remove redundant blocks
            new_current = []
            for bi in current:
                is_redundant = True
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in block_t_indices[bi]:
                        cover_count[idx] -= 1
                    selected_set.remove(bi)
                    improved = True
                else:
                    new_current.append(bi)
            current = new_current
            
            if time.time() > deadline - 10:
                break
                
            # Phase 2: 2-1 Exchange (remove 2 blocks, add 1 if needed)
            n = len(current)
            if n < 2:
                continue
                
            # Build a list of (block_idx, its_t_indices) for quick access
            # We'll iterate over pairs in random order
            pair_list = []
            for i in range(n):
                for j in range(i+1, n):
                    pair_list.append((i, j))
            random.shuffle(pair_list)
            
            found_improvement = False
            for i, j in pair_list:
                if time.time() > deadline - 10:
                    break
                if i >= len(current) or j >= len(current):
                    continue
                
                bi = current[i]
                bj = current[j]
                
                # Compute the union of t-subsets covered by bi and bj
                # mask_bi | mask_bj gives us all t-subsets they cover together
                # The t-subsets that become uncovered are those where cover_count == 1 (for bi) 
                # or cover_count == 1 (for bj) or cover_count == 2 and both cover it
                # More precisely: a t-subset idx is uncovered after removing bi,bj iff
                # cover_count[idx] - (1 if idx in bi else 0) - (1 if idx in bj else 0) == 0
                
                # Efficient approach: 
                # Let union_mask = block_masks[bi] | block_masks[bj]
                # For each idx in union_mask, check if it's still covered
                # But we need to be careful about overlaps.
                
                # Alternative: decrement counts, check, then restore
                # This is O(|bi| + |bj|) which is ~35+35 = 70 per pair
                # With ~10000 pairs, that's 700k operations, which is fine.
                
                for idx in block_t_indices[bi]:
                    cover_count[idx] -= 1
                for idx in block_t_indices[bj]:
                    cover_count[idx] -= 1
                
                # Check if any t-subset became uncovered
                uncovered_indices = []
                # Check indices in bi
                for idx in block_t_indices[bi]:
                    if cover_count[idx] == 0:
                        uncovered_indices.append(idx)
                # Check indices in bj (avoid duplicates)
                for idx in block_t_indices[bj]:
                    if cover_count[idx] == 0 and idx not in uncovered_indices:
                        uncovered_indices.append(idx)
                
                if not uncovered_indices:
                    # Can remove both blocks
                    current.remove(bj)
                    current.remove(bi)
                    selected_set.remove(bj)
                    selected_set.remove(bi)
                    improved = True
                    found_improvement = True
                    break
                else:
                    # Try to find a block bk that covers all uncovered_indices
                    # and is not bi or bj
                    # Sort uncovered by number of covering blocks (rarity first)
                    uncovered_indices.sort(key=lambda x: len(t_to_blocks[x]))
                    
                    candidates = None
                    for idx in uncovered_indices:
                        blocks_covering = t_to_blocks[idx]
                        if candidates is None:
                            candidates = set(blocks_covering)
                        else:
                            candidates.intersection_update(blocks_covering)
                        if not candidates:
                            break
                    
                    if candidates:
                        # Filter: bk must not be in selected_set (to avoid adding duplicate)
                        # and must not be bi or bj
                        valid = [bk for bk in candidates if bk not in selected_set and bk != bi and bk != bj]
                        
                        if valid:
                            bk = random.choice(valid)
                            
                            # Remove bi, bj
                            current.remove(bj)
                            current.remove(bi)
                            selected_set.remove(bj)
                            selected_set.remove(bi)
                            
                            # Add bk
                            current.append(bk)
                            selected_set.add(bk)
                            
                            # Update cover_count: we already decremented bi, bj
                            # Now increment for bk
                            for idx in block_t_indices[bk]:
                                cover_count[idx] += 1
                            
                            improved = True
                            found_improvement = True
                            break
                    else:
                        # No valid replacement, restore counts
                        for idx in block_t_indices[bi]:
                            cover_count[idx] += 1
                        for idx in block_t_indices[bj]:
                            cover_count[idx] += 1
                
            if found_improvement:
                continue
        
        return current

    # Main Loop: Multiple Restarts with diverse weightings
    max_restarts = 15
    weight_modes = ['uniform', 'inverse_freq', 'inverse_sqrt']
    
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Cycle through weight modes for diversity
        wm = weight_modes[r % len(weight_modes)]
        
        # Greedy with randomization
        selected = greedy_construct(randomize=True, weight_mode=wm)
        
        if not selected:
            continue
        
        # Local Search
        selected = local_search(selected)
        
        # Verify and update best
        cov = 0
        for bi in selected:
            cov |= block_masks[bi]
        if cov == full_mask and len(selected) < best_count:
            best_count = len(selected)
            best_blocks = list(selected)
            
        # If we have a good solution, spend extra time polishing it
        if best_count <= 152 and time.time() < deadline - 30:
            # Do additional local search passes on the best solution
            polished = local_search(list(best_blocks))
            cov2 = 0
            for bi in polished:
                cov2 |= block_masks[bi]
            if cov2 == full_mask and len(polished) < best_count:
                best_count = len(polished)
                best_blocks = list(polished)

    if best_blocks is None:
        # Fallback
        selected = greedy_construct(randomize=False)
        selected = local_search(selected)
        best_blocks = selected

    return [list(all_k_blocks[i]) for i in best_blocks]
