import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148  # Safety margin
    
    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(points, k))
    num_k = len(k_blocks)
    
    # Map each block to the indices of t-subsets it covers (as numpy arrays for speed)
    block_t_indices = []
    for block in k_blocks:
        covered = set()
        for sub in itertools.combinations(block, t):
            covered.add(t_index[sub])
        block_t_indices.append(np.array(list(covered), dtype=np.int32))
        
    # Map each t-subset to the indices of blocks that cover it
    t_to_blocks = [set() for _ in range(num_t)]
    for b_idx, subs in enumerate(block_t_indices):
        for s_idx in subs:
            t_to_blocks[s_idx].add(b_idx)

    if elapsed() > deadline - 60:
        return [list(k_blocks[0])] 

    # 2. Lagrangian Relaxation with Subgradient Optimization
    # Dynamic weights guide greedy selection toward hard-to-cover subsets
    t_weights = np.ones(num_t, dtype=np.float64)
    
    best_selected = None
    best_size = float('inf')
    
    # Phase 1: Multiple greedy passes with evolving weights
    num_greedy_rounds = 5
    for round_idx in range(num_greedy_rounds):
        if elapsed() > deadline - 30:
            break
        
        # Reset weights based on current difficulty
        # First round: uniform; later rounds: boost under-covered subsets
        if round_idx == 0:
            t_weights[:] = 1.0
        else:
            # Count how many times each t-subset was covered in previous best solution
            if best_selected is not None:
                cover_hist = np.zeros(num_t, dtype=np.int32)
                for b_idx in best_selected:
                    for s_idx in block_t_indices[b_idx]:
                        cover_hist[s_idx] += 1
                # Subsets covered few times get higher weight
                t_weights[:] = 1.0 / (cover_hist + 1.0)
                t_weights /= np.sum(t_weights) * 1e-6  # Normalize
        
        # Greedy selection with current weights
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        
        # Compute initial gains
        gain = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if len(block_t_indices[i]) > 0:
                gain[i] = np.sum(t_weights[block_t_indices[i]])
        
        while uncovered_count > 0 and elapsed() < deadline - 25:
            current_gains = gain.copy()
            current_gains[selected_mask] = -1.0
            best_idx = int(np.argmax(current_gains))
            
            if gain[best_idx] <= 0:
                break
            
            selected.append(best_idx)
            selected_mask[best_idx] = True
            
            for idx in block_t_indices[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w
        
        # Track best solution found so far
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
    
    # Use the best greedy solution as starting point
    if best_selected is None:
        # Fallback: simple greedy
        t_weights[:] = 1.0
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        gain = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if len(block_t_indices[i]) > 0:
                gain[i] = np.sum(t_weights[block_t_indices[i]])
        while uncovered_count > 0 and elapsed() < deadline - 20:
            current_gains = gain.copy()
            current_gains[selected_mask] = -1.0
            best_idx = int(np.argmax(current_gains))
            if gain[best_idx] <= 0:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True
            for idx in block_t_indices[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            gain[b] -= w
        best_selected = selected
    
    selected = list(best_selected)
    
    # 3. Remove Redundant Blocks
    cover_count = np.zeros(num_t, dtype=np.int32)
    for b_idx in selected:
        for s_idx in block_t_indices[b_idx]:
            cover_count[s_idx] += 1
            
    changed = True
    while changed and elapsed() < deadline - 15:
        changed = False
        new_selected = []
        for b_idx in selected:
            needed = False
            for s_idx in block_t_indices[b_idx]:
                if cover_count[s_idx] == 1:
                    needed = True
                    break
            if needed:
                new_selected.append(b_idx)
            else:
                changed = True
                for s_idx in block_t_indices[b_idx]:
                    cover_count[s_idx] -= 1
        if changed:
            selected = new_selected

    # 4. Iterated Local Search with Perturbation
    # This is the key structural change: ILS can escape local minima
    def try_drop_blocks():
        nonlocal selected, cover_count
        changed = True
        while changed and elapsed() < deadline - 8:
            changed = False
            new_selected = []
            for b_idx in selected:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cover_count[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cover_count[s_idx] -= 1
            if changed:
                selected = new_selected
                return True
        return False

    def compute_gain_array():
        """Compute gain for all blocks given current cover_count."""
        g = np.zeros(num_k, dtype=np.float64)
        for i in range(num_k):
            if len(block_t_indices[i]) == 0:
                continue
            # Gain = number of newly covered subsets
            # A subset is "newly covered" if cover_count[s] == 0
            # But we also want to avoid adding blocks that don't help
            g[i] = np.sum(cover_count[block_t_indices[i]] == 0)
        return g

    def greedy_repair():
        """Add blocks to cover all uncovered t-subsets."""
        nonlocal selected, cover_count
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        for b_idx in selected:
            selected_mask[b_idx] = True
        
        while elapsed() < deadline - 5:
            # Find uncovered t-subsets
            uncovered_idx = np.where(cover_count == 0)[0]
            if len(uncovered_idx) == 0:
                break
            
            # Pick the most constrained uncovered subset
            # (fewest candidate blocks)
            min_count = float('inf')
            best_uncovered = uncovered_idx[0]
            for s_idx in uncovered_idx[:50]:  # Sample for speed
                cnt = len(t_to_blocks[s_idx])
                if cnt < min_count:
                    min_count = cnt
                    best_uncovered = s_idx
            
            # Get candidate blocks covering best_uncovered
            candidates = list(t_to_blocks[best_uncovered])
            if not candidates:
                break
            
            # Pick candidate with highest gain
            best_cand = candidates[0]
            best_gain = -1
            for b in candidates:
                if selected_mask[b]:
                    continue
                g = 0
                for s_idx in block_t_indices[b]:
                    if cover_count[s_idx] == 0:
                        g += 1
                if g > best_gain:
                    best_gain = g
                    best_cand = b
            
            if best_gain <= 0:
                break
            
            selected.append(best_cand)
            selected_mask[best_cand] = True
            for s_idx in block_t_indices[best_cand]:
                cover_count[s_idx] += 1
        
        # Final drop pass
        try_drop_blocks()

    def local_search_round():
        """Perform swaps and drops to improve the solution."""
        nonlocal selected, cover_count
        improved = True
        while improved and elapsed() < deadline - 5:
            improved = False
            if try_drop_blocks():
                improved = True
                continue
            
            # Try 1-for-1 swaps
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
                
                cand_list = list(candidates)
                if len(cand_list) > 15:
                    random.shuffle(cand_list)
                    cand_list = cand_list[:15]
                
                # Score candidates by how much they help (cover other unique subsets of other blocks)
                scored = []
                for b_in in cand_list:
                    score = 0
                    for s_idx in block_t_indices[b_in]:
                        if cover_count[s_idx] >= 2:
                            score += 1
                    scored.append((score, b_in))
                scored.sort(reverse=True)
                
                for _, b_in in scored:
                    if elapsed() > deadline - 5:
                        break
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] += 1
                    selected[i] = b_in
                    
                    if try_drop_blocks():
                        improved = True
                        break
                    else:
                        for s_idx in block_t_indices[b_in]:
                            cover_count[s_idx] -= 1
                        for s_idx in block_t_indices[b_out]:
                            cover_count[s_idx] += 1
                        selected[i] = b_out
                
                if improved:
                    break
        
        return len(selected)

    # ILS loop: perturb, repair, local search
    best_ils_size = len(selected)
    best_ils_selected = list(selected)
    best_ils_cover = cover_count.copy()
    
    num_ils_iterations = 8
    for ils_iter in range(num_ils_iterations):
        if elapsed() > deadline - 10:
            break
        
        # Perturbation: randomly remove some blocks
        if len(selected) > 10:
            num_remove = max(1, len(selected) // 20)
            remove_idx = random.sample(range(len(selected)), num_remove)
            for idx in sorted(remove_idx, reverse=True):
                b_out = selected[idx]
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                selected.pop(idx)
        
        # Repair: add blocks to cover all uncovered subsets
        greedy_repair()
        
        # Local search
        local_search_round()
        
        # Track best
        if len(selected) < best_ils_size:
            best_ils_size = len(selected)
            best_ils_selected = list(selected)
            best_ils_cover = cover_count.copy()
    
    # Restore best ILS solution
    if best_ils_size < len(selected):
        selected = list(best_ils_selected)
        cover_count = best_ils_cover.copy()
    
    # 5. Final polish
    if elapsed() < deadline - 2:
        improved2 = True
        while improved2 and elapsed() < deadline - 2:
            improved2 = False
            if try_drop_blocks():
                improved2 = True
                continue
            for i in range(len(selected)):
                if elapsed() > deadline - 2:
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
                b_in = next(iter(candidates))
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                selected[i] = b_in
                if try_drop_blocks():
                    improved2 = True
                    break
                else:
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out

    # Final result
    result = [list(k_blocks[i]) for i in selected]
    return result
