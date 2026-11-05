import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    # Compute masks: for each block, which t-subsets does it cover?
    block_data = []
    block_masks = []
    for block in all_k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_data.append(block)
        block_masks.append(mask)
    
    full_mask = (1 << n_t) - 1
    
    # Precompute bit counts for all block masks
    block_bit_counts = [m.bit_count() for m in block_masks]
    
    # Precompute for each t-subset, the list of blocks that cover it
    t_to_blocks = [[] for _ in range(n_t)]
    for i, m in enumerate(block_masks):
        mm = m
        while mm:
            lsb = mm & -mm
            idx = lsb.bit_length() - 1
            t_to_blocks[idx].append(i)
            mm ^= lsb
    
    # Greedy set cover with a scoring function
    def greedy_construct(score_func=None, seed=0):
        """Greedy set cover. score_func(gain, block_idx, uncovered_mask) -> score"""
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        # Track remaining blocks
        remaining = list(range(n_k))
        
        while uncovered != 0:
            if time.time() > deadline - 20:
                break
            best_idx = -1
            best_score = -1
            
            # To speed up, we can sample or use a smarter heuristic
            # For now, scan all remaining
            for i in remaining:
                gain = (block_masks[i] & uncovered).bit_count()
                if gain == 0:
                    continue
                if score_func:
                    score = score_func(gain, i, uncovered, block_bit_counts[i])
                else:
                    score = gain
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
            remaining.remove(best_idx)
        
        return selected
    
    # Prune redundant blocks
    def prune_blocks(selected_indices):
        if not selected_indices:
            return selected_indices
        
        current = selected_indices[:]
        current_set = set(current)
        
        # Build cover counts
        cover_count = [0] * n_t
        for idx in current:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                i = lsb.bit_length() - 1
                cover_count[i] += 1
                m ^= lsb
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            new_current = []
            for idx in current:
                mask = block_masks[idx]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    i = lsb.bit_length() - 1
                    if cover_count[i] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                
                if is_critical:
                    new_current.append(idx)
                else:
                    m = mask
                    while m:
                        lsb = m & -m
                        i = lsb.bit_length() - 1
                        cover_count[i] -= 1
                        m ^= lsb
                    current_set.discard(idx)
                    improved = True
            
            current = new_current
        
        return current
    
    # Local search: 1-swap (remove one, add one)
    def local_search(selected_indices, max_time=30):
        if not selected_indices:
            return selected_indices
        
        current = prune_blocks(selected_indices)
        current_set = set(current)
        best = current[:]
        best_len = len(best)
        
        # Build coverage counts
        cov = [0] * n_t
        for idx in current:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb
        
        # Try deletions first
        for i in range(len(current)):
            if time.time() > deadline - 15:
                break
            removed_idx = current[i]
            mask = block_masks[removed_idx]
            
            can_remove = True
            m = mask
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                if cov[k2] == 1:
                    can_remove = False
                    break
                m ^= lsb
            
            if can_remove:
                m = mask
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    cov[k2] -= 1
                    m ^= lsb
                current.pop(i)
                current_set.discard(removed_idx)
                best = current[:]
                best_len = len(best)
                i -= 1
        
        # Try swaps
        if time.time() < deadline - 20:
            # Precompute gain for each block against current uncovered (if any)
            # But we want to swap: remove block i, add block j
            # The gain of j is the number of t-subsets it covers that are ONLY covered by i (or uncovered after removing i)
            
            for i in range(len(current)):
                if time.time() > deadline - 15:
                    break
                removed_idx = current[i]
                mask = block_masks[removed_idx]
                
                # Temporarily remove
                m = mask
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    cov[k2] -= 1
                    m ^= lsb
                
                # Find t-subsets that are now uncovered (cov == 0)
                uncovered_mask = 0
                for k2 in range(n_t):
                    if cov[k2] == 0:
                        uncovered_mask |= (1 << k2)
                
                if uncovered_mask == 0:
                    # Just remove
                    current.pop(i)
                    current_set.discard(removed_idx)
                    best = current[:]
                    best_len = len(best)
                    i -= 1
                else:
                    # Find best replacement
                    best_add = -1
                    best_gain = -1
                    target = uncovered_mask.bit_count()
                    
                    for j in range(n_k):
                        if j in current_set:
                            continue
                        gain = (block_masks[j] & uncovered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_add = j
                        if gain == target:
                            break
                    
                    if best_add >= 0 and best_gain > 0:
                        current[i] = best_add
                        current_set.discard(removed_idx)
                        current_set.add(best_add)
                        m = block_masks[best_add]
                        while m:
                            lsb = m & -m
                            k2 = lsb.bit_length() - 1
                            cov[k2] += 1
                            m ^= lsb
                        best = current[:]
                        best_len = len(best)
                    else:
                        # Restore
                        m = mask
                        while m:
                            lsb = m & -m
                            k2 = lsb.bit_length() - 1
                            cov[k2] += 1
                            m ^= lsb
        
        return prune_blocks(best)
    
    # Phase 1: Greedy with different scoring functions
    best_solution = None
    best_len = float('inf')
    
    # 1. Standard greedy (max gain)
    sol = greedy_construct(seed=0)
    sol = prune_blocks(sol)
    sol = local_search(sol)
    best_solution = sol
    best_len = len(sol)
    
    # 2. Greedy with penalty for overlap
    def score_penalty(gain, idx, unc, bit_count):
        overlap = bit_count - gain
        return gain / (1 + 0.1 * overlap)
    
    if time.time() < deadline - 40:
        sol = greedy_construct(score_func=score_penalty, seed=1)
        sol = prune_blocks(sol)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # 3. Greedy prioritizing rare subsets
    # Precompute frequency of each t-subset
    sub_freq = [0] * n_t
    for idx in range(n_k):
        m = block_masks[idx]
        while m:
            lsb = m & -m
            k2 = lsb.bit_length() - 1
            sub_freq[k2] += 1
            m ^= lsb
    
    def score_rare(gain, idx, unc, bit_count):
        # We need the actual gain mask to compute weighted score
        # This is a simplified version
        return gain
    
    if time.time() < deadline - 40:
        # Custom greedy for rare subsets
        uncovered = full_mask
        selected = []
        remaining = list(range(n_k))
        
        while uncovered != 0:
            if time.time() > deadline - 30:
                break
            best_idx = -1
            best_score = -1.0
            for i in remaining:
                gain_mask = block_masks[i] & uncovered
                gain = gain_mask.bit_count()
                if gain == 0:
                    continue
                score = 0.0
                m = gain_mask
                while m:
                    lsb = m & -m
                    k2 = lsb.bit_length() - 1
                    score += 1.0 / (1 + 0.01 * sub_freq[k2])
                    m ^= lsb
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            remaining.remove(best_idx)
        
        sol = prune_blocks(selected)
        sol = local_search(sol)
        if len(sol) < best_len:
            best_solution = sol
            best_len = len(sol)
    
    # Phase 2: ILS
    rng = random.Random(42)
    ils_iter = 0
    while time.time() < deadline - 20 and ils_iter < 100:
        ils_iter += 1
        
        perturbed = best_solution[:]
        perturbed_set = set(perturbed)
        
        # Perturbation: remove 1-2 blocks
        if len(perturbed) > 2:
            n_remove = min(rng.randint(1, 2), len(perturbed) - 1)
            to_remove = rng.sample(range(len(perturbed)), n_remove)
            to_remove_set = set(to_remove)
            for i in sorted(to_remove_set, reverse=True):
                perturbed_set.discard(perturbed[i])
                perturbed.pop(i)
        
        # Re-optimize
        cov = [0] * n_t
        for idx in perturbed:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                k2 = lsb.bit_length() - 1
                cov[k2] += 1
                m ^= lsb
        
        uncovered = 0
        for k2 in range(n_t):
            if cov[k2] == 0:
                uncovered |= (1 << k2)
        
        while uncovered != 0 and time.time() < deadline - 15:
            best_idx = -1
            best_gain = -1
            for j in range(n_k):
                if j in perturbed_set:
                    continue
                gain = (block_masks[j] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            if best_idx == -1 or best_gain == 0:
                break
            perturbed.append(best_idx)
            perturbed_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        perturbed = prune_blocks(perturbed)
        perturbed = local_search(perturbed)
        
        if len(perturbed) < best_len:
            best_solution = perturbed
            best_len = len(perturbed)
    
    # Final validation
    final_cov = [0] * n_t
    for idx in best_solution:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            k2 = lsb.bit_length() - 1
            final_cov[k2] += 1
            m ^= lsb
    
    sol_set = set(best_solution)
    for k2 in range(n_t):
        if final_cov[k2] == 0:
            for j in range(n_k):
                if (block_masks[j] >> k2) & 1:
                    if j not in sol_set:
                        best_solution.append(j)
                        sol_set.add(j)
                    break
    
    blocks = [list(block_data[i]) for i in best_solution]
    
    # Ensure distinct
    seen = set()
    unique_blocks = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)
    
    return unique_blocks
