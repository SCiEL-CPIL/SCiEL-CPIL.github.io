import itertools
import time
import random
import numpy as np
import math

def _remove_redundant(blocks_idx, block_masks, n_t_subs, deadline):
    """Remove redundant blocks (1-exchange)."""
    if not blocks_idx:
        return blocks_idx
    
    current = list(blocks_idx)
    
    # Build cover count
    cover_count = [0] * n_t_subs
    for bi in current:
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        new_current = []
        for i in range(len(current)):
            bi = current[i]
            mask = block_masks[bi]
            is_critical = False
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_critical = True
                    break
                m ^= lsb
            if is_critical:
                new_current.append(bi)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
        current = new_current
    
    return current

def _two_exchange(blocks_idx, block_masks, all_k_blocks, n_k_blocks, n_t_subs, deadline):
    """Try 2-exchanges: remove one block, try to add one different block that covers more."""
    if not blocks_idx:
        return blocks_idx
    
    current = list(blocks_idx)
    full_mask = (1 << n_t_subs) - 1
    
    # Build coverage mask for current set
    coverage = 0
    for bi in current:
        coverage |= block_masks[bi]
    
    # Build cover count
    cover_count = [0] * n_t_subs
    for bi in current:
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        n = len(current)
        for i in range(n):
            if time.time() > deadline - 10:
                break
            bi = current[i]
            removed_mask = block_masks[bi]
            
            # Check if coverage is still complete without this block
            new_coverage = 0
            for j in range(n):
                if j != i:
                    new_coverage |= block_masks[current[j]]
            
            # Find uncovered subsets
            uncovered = ~new_coverage & full_mask
            if uncovered == 0:
                # Can simply remove this block
                new_current = [current[j] for j in range(n) if j != i]
                current = new_current
                # Rebuild cover_count
                cover_count = [0] * n_t_subs
                for bi2 in current:
                    m = block_masks[bi2]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] += 1
                        m ^= lsb
                improved = True
                break
            else:
                # Try to find a block that covers the most uncovered subsets
                best_gain = 0
                best_block = -1
                for bi2 in range(n_k_blocks):
                    if bi2 in current:
                        continue
                    gain = (block_masks[bi2] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = bi2
                    if best_gain >= uncovered.bit_count():
                        break
                
                if best_block != -1 and best_gain > 0:
                    # Check: after adding best_block, is everything covered?
                    if (uncovered & ~block_masks[best_block]) == 0:
                        # Replace bi with best_block
                        new_current = list(current)
                        new_current[i] = best_block
                        current = new_current
                        # Rebuild cover_count
                        cover_count = [0] * n_t_subs
                        for bi2 in current:
                            m = block_masks[bi2]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_count[idx] += 1
                                m ^= lsb
                        improved = True
                        break
    
    return current

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_k_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    full_mask = (1 << n_t_subs) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different greedy orderings
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        # Greedy construction with random tie-breaking
        uncovered_mask = full_mask
        selected = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 30:
                break
            
            best_idx = -1
            best_gain = -1
            best_gain_list = []
            
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_gain_list = [i]
                elif gain == best_gain:
                    best_gain_list.append(i)
            
            if best_idx == -1 or best_gain == 0:
                break
            
            # Random tie-breaking for diversity
            if len(best_gain_list) > 1:
                best_idx = random.choice(best_gain_list)
            
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        
        # Local search: redundancy removal
        selected = _remove_redundant(selected, block_masks, n_t_subs, deadline)
        
        # Local search: 2-exchange
        selected = _two_exchange(selected, block_masks, all_k_blocks, n_k_blocks, n_t_subs, deadline)
        
        # Improved Simulated Annealing with adaptive temperature and multiple moves
        if time.time() < deadline - 15:
            sa_time_limit = deadline - 15
            current_blocks = list(selected)
            
            # Build cover count for SA
            cover_count = [0] * n_t_subs
            for bi in current_blocks:
                m = block_masks[bi]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] += 1
                    m ^= lsb
            
            # Temperature schedule
            T_start = 10.0
            T_end = 0.05
            sa_iterations = 0
            max_sa_iters = 500
            no_improve_count = 0
            best_sa_count = len(current_blocks)
            
            while time.time() < sa_time_limit and sa_iterations < max_sa_iters:
                sa_iterations += 1
                progress = sa_iterations / max_sa_iters
                T = T_start * (T_end / T_start) ** progress
                
                # Adaptive: if stuck, increase temperature slightly
                if no_improve_count > 50:
                    T = min(T * 1.5, 20.0)
                    no_improve_count = 0
                
                if len(current_blocks) == 0:
                    break
                
                # Random move: either remove a block or swap a block
                move_type = random.random()
                
                if move_type < 0.3:
                    # Try to remove a block
                    remove_idx = random.randint(0, len(current_blocks) - 1)
                    removed_block = current_blocks[remove_idx]
                    
                    # Check if we can simply remove
                    new_cover_count = cover_count[:]
                    m = block_masks[removed_block]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        new_cover_count[idx] -= 1
                        m ^= lsb
                    
                    # Check if any t-subset becomes uncovered
                    can_remove = True
                    m = block_masks[removed_block]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if new_cover_count[idx] == 0:
                            can_remove = False
                            break
                        m ^= lsb
                    
                    if can_remove:
                        # Remove the block
                        current_blocks.pop(remove_idx)
                        cover_count = new_cover_count
                        no_improve_count = 0
                        if len(current_blocks) < best_sa_count:
                            best_sa_count = len(current_blocks)
                    else:
                        no_improve_count += 1
                        
                else:
                    # Swap a block with another
                    swap_idx = random.randint(0, len(current_blocks) - 1)
                    old_block = current_blocks[swap_idx]
                    
                    # Find best replacement
                    best_gain = 0
                    best_block = -1
                    for bi in range(n_k_blocks):
                        if bi in current_blocks:
                            continue
                        gain = (block_masks[bi] & block_masks[old_block]).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_block = bi
                    
                    if best_block == -1:
                        no_improve_count += 1
                        continue
                    
                    # Calculate energy change
                    # Old uncovered = t-subsets only covered by old_block
                    # New uncovered = t-subsets only covered by new_block
                    old_only_count = 0
                    new_only_count = 0
                    m = block_masks[old_block]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            old_only_count += 1
                        m ^= lsb
                    
                    m = block_masks[best_block]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        # After swap, cover_count changes
                        # New count for idx = cover_count[idx] - (1 if idx in old_block else 0) + (1 if idx in new_block else 0)
                        old_has = 1 if (block_masks[old_block] & (1 << idx)) else 0
                        new_has = 1 if (block_masks[best_block] & (1 << idx)) else 0
                        new_count = cover_count[idx] - old_has + new_has
                        if new_count == 0:
                            new_only_count += 1
                        m ^= lsb
                    
                    delta_energy = new_only_count - old_only_count
                    
                    # Accept if improvement or with probability exp(-delta/T)
                    if delta_energy <= 0:
                        # Accept improvement
                        current_blocks[swap_idx] = best_block
                        # Update cover_count
                        m = block_masks[old_block]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        m = block_masks[best_block]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            m ^= lsb
                        no_improve_count = 0
                        if len(current_blocks) < best_sa_count:
                            best_sa_count = len(current_blocks)
                    else:
                        # Accept worse solution with probability exp(-delta/T)
                        if random.random() < math.exp(-delta_energy / T):
                            current_blocks[swap_idx] = best_block
                            # Update cover_count
                            m = block_masks[old_block]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_count[idx] -= 1
                                m ^= lsb
                            m = block_masks[best_block]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                cover_count[idx] += 1
                                m ^= lsb
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
            
            # Verify coverage after SA
            final_cov = 0
            for bi in current_blocks:
                final_cov |= block_masks[bi]
            if final_cov == full_mask and len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = list(current_blocks)
        
        # Verify coverage
        cov = 0
        for bi in selected:
            cov |= block_masks[bi]
        if cov == full_mask and len(selected) < best_count:
            best_count = len(selected)
            best_blocks = list(selected)
        
        if time.time() > deadline - 15:
            break
    
    if best_blocks is None:
        # Fallback: greedy without restarts
        uncovered_mask = full_mask
        selected = []
        while uncovered_mask != 0:
            if time.time() > deadline - 5:
                break
            best_idx = -1
            best_gain = -1
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        selected = _remove_redundant(selected, block_masks, n_t_subs, deadline)
        best_blocks = selected
    
    return [list(all_k_blocks[i]) for i in best_blocks]
