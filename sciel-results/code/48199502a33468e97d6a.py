import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(points, k))
    num_k = len(k_blocks)
    
    # Map each block to the indices of t-subsets it covers
    block_t_indices = []
    for block in k_blocks:
        covered = set()
        for sub in itertools.combinations(block, t):
            covered.add(t_index[sub])
        block_t_indices.append(list(covered))
        
    # Map each t-subset to the indices of blocks that cover it
    t_to_blocks = [set() for _ in range(num_t)]
    for b_idx, subs in enumerate(block_t_indices):
        for s_idx in subs:
            t_to_blocks[s_idx].add(b_idx)

    if elapsed() > deadline - 60:
        return [list(k_blocks[0])] 

    # 2. Dual-guided Primal-Dual Set Cover
    def primal_dual_greedy(seed_val):
        rng = random.Random(seed_val)
        
        # Initialize dual variables: start with inverse frequency
        dual = np.zeros(num_t, dtype=np.float64)
        for i in range(num_t):
            count = len(t_to_blocks[i])
            if count > 0:
                dual[i] = 1.0 / count
            else:
                dual[i] = 0.0
        
        # Compute initial gain for each block (dual-weighted coverage)
        gain = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if block_t_indices[i]:
                gain[i] = np.sum(dual[block_t_indices[i]])
                
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        
        # Primal-dual: iteratively raise duals of uncovered sets, select when a block becomes "tight"
        alpha = 1.0  # scaling factor for dual updates
        beta = 0.5   # decay factor for dual updates
        
        while uncovered_count > 0 and elapsed() < deadline - 40:
            # Find best block by dual-weighted gain
            current_gains = gain.copy()
            current_gains[selected_mask] = -1.0
            best_idx = np.argmax(current_gains)
            
            if gain[best_idx] <= 0:
                break
            
            # Random tie-breaking for diversity
            threshold = gain[best_idx] * 0.98
            candidates = np.where((current_gains >= threshold) & (~selected_mask))[0]
            if len(candidates) > 1:
                best_idx = candidates[rng.randrange(len(candidates))]
                
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update: mark covered sets, adjust duals
            newly_covered = []
            for idx in block_t_indices[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    newly_covered.append(idx)
            
            # Dual update: reduce duals of newly covered sets (they're now satisfied)
            # and slightly increase duals of still-uncovered sets that could be covered by remaining blocks
            for idx in newly_covered:
                old_dual = dual[idx]
                dual[idx] = 0.0
                # Update gains: subtract old dual from all blocks covering this set
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= old_dual
            
            # Periodically boost duals of hard-to-cover uncovered sets
            if len(selected) % 10 == 0 and uncovered_count > 0:
                # Find uncovered sets with fewest covering blocks
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) > 0:
                    # Sort by scarcity (fewest blocks cover them)
                    scarcity = np.array([len(t_to_blocks[i]) for i in uncovered_indices])
                    # Boost the most scarce ones
                    for idx in uncovered_indices[np.argsort(scarcity)[:min(50, len(uncovered_indices))]]:
                        if len(t_to_blocks[idx]) > 0:
                            boost = alpha / (len(t_to_blocks[idx]) + 1)
                            old_dual = dual[idx]
                            dual[idx] = old_dual + boost * beta
                            # Update gains for blocks covering this set
                            for b in t_to_blocks[idx]:
                                if not selected_mask[b]:
                                    gain[b] += boost * beta
        
        return selected

    # Run multiple primal-dual restarts
    best_selected = None
    best_size = float('inf')
    num_restarts = 10
    restart_time_budget = deadline - 70  # Leave time for local search
    
    for r in range(num_restarts):
        if elapsed() > restart_time_budget:
            break
        selected = primal_dual_greedy(r * 12345 + 67890)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]

    if best_selected is None:
        best_selected = primal_dual_greedy(0)
        best_size = len(best_selected)

    # 3. Remove Redundant Blocks
    selected = best_selected[:]
    
    def compute_cover_count(sel):
        cc = np.zeros(num_t, dtype=np.int32)
        for b_idx in sel:
            for s_idx in block_t_indices[b_idx]:
                cc[s_idx] += 1
        return cc

    cover_count = compute_cover_count(selected)
    
    def try_drop_blocks(sel, cc):
        changed = True
        while changed and elapsed() < deadline - 10:
            changed = False
            new_selected = []
            for b_idx in sel:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cc[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cc[s_idx] -= 1
            if changed:
                sel = new_selected
        return sel, cc

    selected, cover_count = try_drop_blocks(selected, cover_count)

    # 4. Local Search: 1-1 Swaps and 2-1 Swaps
    def try_drop_blocks_local(sel, cc):
        changed = True
        while changed and elapsed() < deadline - 5:
            changed = False
            new_selected = []
            for b_idx in sel:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cc[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cc[s_idx] -= 1
            if changed:
                sel = new_selected
        return sel, cc

    improved = True
    while improved and elapsed() < deadline - 5:
        improved = False
        
        # Try to drop first
        selected, cover_count = try_drop_blocks_local(selected, cover_count)
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
            improved = True
            continue
            
        # Try 1-1 swaps with dual-guided candidate selection
        for i in range(len(selected)):
            if elapsed() > deadline - 5:
                break
            b_out = selected[i]
            unique_subs = [s_idx for s_idx in block_t_indices[b_out] if cover_count[s_idx] == 1]
            
            if not unique_subs:
                continue
                
            candidates = None
            for s_idx in unique_subs:
                if candidates is None:
                    candidates = set(t_to_blocks[s_idx])
                else:
                    candidates &= t_to_blocks[s_idx]
                    
            if not candidates:
                continue
                
            candidates.discard(b_out)
            if not candidates:
                continue
                
            # Score candidates by how many additional uncovered sets they cover
            # (sets that are only covered by b_out and this candidate)
            candidate_scores = []
            for b_in in candidates:
                # Count how many unique-to-b_out sets this candidate also covers
                overlap = 0
                for s_idx in unique_subs:
                    if b_in in t_to_blocks[s_idx]:
                        overlap += 1
                # Also count new coverage
                new_cov = 0
                for s_idx in block_t_indices[b_in]:
                    if cover_count[s_idx] == 0:
                        new_cov += 1
                score = overlap * 2 + new_cov  # Prioritize covering the critical sets
                candidate_scores.append((score, b_in))
            
            # Sort by score, try top candidates
            candidate_scores.sort(reverse=True)
            
            for _, b_in in candidate_scores[:8]:  # Limit to 8 candidates per position
                # Perform swap
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                    
                selected[i] = b_in
                
                new_sel, new_cc = try_drop_blocks_local(selected, cover_count)
                
                if len(new_sel) < len(selected):
                    selected, cover_count = new_sel, new_cc
                    best_size = len(selected)
                    best_selected = selected[:]
                    improved = True
                    break
                else:
                    # Revert
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out

    # Final result
    result = [list(k_blocks[i]) for i in best_selected]
    return result
