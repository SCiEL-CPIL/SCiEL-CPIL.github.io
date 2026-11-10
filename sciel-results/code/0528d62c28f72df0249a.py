import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer

    # 1. Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(points, k))
    n_k = len(all_k_subsets)
    
    # masks[i] is a bitmask of triples covered by block i
    # t_indices[i] is a list of triple indices covered by block i
    masks = [0] * n_k
    t_indices = [None] * n_k
    
    for i, k_sub in enumerate(all_k_subsets):
        m = 0
        idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_to_idx[t_sub]
            m |= (1 << idx)
            idxs.append(idx)
        masks[i] = m
        t_indices[i] = idxs
        
    full_mask = (1 << n_t) - 1
    
    def greedy_cover(uncovered_mask, rng):
        selected = []
        current_uncovered = uncovered_mask
        # We need to find blocks that cover parts of current_uncovered
        # To speed up, we can just iterate all blocks. 
        # Optimization: If we maintain a set of "candidates", it's complex.
        # Let's just iterate all n_k blocks.
        
        while current_uncovered:
            best_gain = -1
            best_idx = -1
            best_gain_list = []
            
            # Iterate all blocks
            for i in range(n_k):
                # Skip if already selected? 
                # We don't strictly need to skip, but it saves time if we track selected set.
                # However, a block might cover *more* uncovered triples than another.
                # If a block is selected, its contribution to *further* uncovered is 0 
                # only if we update current_uncovered. Yes, we do.
                # So selected blocks will have gain 0 (or close to it if they overlap with previously selected in a way that doesn't reduce uncovered? No, current_uncovered is reduced by selected blocks).
                # So selected blocks will have (masks[i] & current_uncovered) == 0.
                # We can skip them if we keep a set.
                
                # Let's use a set for selected indices to skip them.
                # But checking `i in selected_set` is O(1).
                
                gain = (masks[i] & current_uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_gain_list = [i]
                elif gain == best_gain:
                    best_gain_list.append(i)
                    
            if best_gain == 0:
                break # Should not happen if valid cover exists and we didn't break early
                
            # Random tie-breaking
            best_idx = best_gain_list[rng.randint(0, len(best_gain_list) - 1)]
            selected.append(best_idx)
            current_uncovered &= ~masks[best_idx]
            
            # Time check
            if time.time() > deadline - 5:
                break
                
        return selected

    def reduce_cover(blocks):
        if not blocks:
            return []
            
        current_blocks = list(blocks)
        # Count coverage for each triple
        # n_t is 2024.
        cov_counts = [0] * n_t
        
        for b_idx in current_blocks:
            for idx in t_indices[b_idx]:
                cov_counts[idx] += 1
                
        improved = True
        rng_reduce = random.Random(12345)
        
        while improved:
            improved = False
            # Shuffle to avoid bias
            order = list(range(len(current_blocks)))
            rng_reduce.shuffle(order)
            
            for bi in order:
                if bi >= len(current_blocks):
                    continue
                    
                # Check if block bi is redundant
                # It is redundant if all its triples are covered by other blocks
                # i.e., cov_counts[idx] > 1 for all idx in t_indices[bi]
                redundant = True
                for idx in t_indices[current_blocks[bi]]:
                    if cov_counts[idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove it
                    for idx in t_indices[current_blocks[bi]]:
                        cov_counts[idx] -= 1
                    current_blocks.pop(bi)
                    improved = True
                    break
                    
        return current_blocks

    def is_valid_cover(blocks):
        covered = 0
        for b_idx in blocks:
            covered |= masks[b_idx]
        return covered == full_mask

    # Main ILS
    rng_main = random.Random(42)
    best_blocks = None
    best_len = float('inf')
    
    # Initial Greedy
    for restart in range(10):
        if time.time() > deadline - 20:
            break
        rng_init = random.Random(42 + restart)
        init_sel = greedy_cover(full_mask, rng_init)
        reduced = reduce_cover(init_sel)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_blocks = reduced
            
    # ILS Loop
    no_improve = 0
    while time.time() < deadline - 10:
        current_blocks = list(best_blocks)
        
        # Perturbation: Remove random blocks
        # Number to remove: 1 to 3
        n_remove = rng_main.randint(1, 3)
        if len(current_blocks) <= n_remove:
            continue
            
        # Select blocks to remove. 
        # Heuristic: Remove blocks with low "unique" contribution?
        # Or just random? Random is simpler and effective in ILS.
        # Let's try random first.
        remove_indices = set(rng_main.sample(range(len(current_blocks)), n_remove))
        
        new_current = []
        removed_mask = 0
        for i, b_idx in enumerate(current_blocks):
            if i in remove_indices:
                removed_mask |= masks[b_idx]
            else:
                new_current.append(b_idx)
                
        # Calculate uncovered
        covered_by_current = 0
        for b_idx in new_current:
            covered_by_current |= masks[b_idx]
            
        uncovered = removed_mask & ~covered_by_current
        
        if uncovered == 0:
            # We removed blocks but coverage didn't drop? Impossible unless redundant.
            # But reduce_cover should have removed redundants.
            # So this means the removed blocks were not needed for *new* uncovered?
            # Wait, if we remove a block, its unique triples become uncovered.
            # If uncovered is 0, it means the removed blocks had no unique triples.
            # This implies they were redundant.
            # So we can just accept the smaller set.
            if len(new_current) < best_len:
                best_len = len(new_current)
                best_blocks = new_current
                no_improve = 0
            continue
            
        # Repair
        repair_sel = greedy_cover(uncovered, rng_main)
        candidate_blocks = new_current + repair_sel
        candidate_blocks = reduce_cover(candidate_blocks)
        
        if len(candidate_blocks) < best_len:
            best_len = len(candidate_blocks)
            best_blocks = candidate_blocks
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                # Re-seed or stronger perturbation?
                # Just reset counter to allow continued search
                no_improve = 0

    # Final Polish: Try to remove 1 block
    if time.time() < deadline - 2:
        for _ in range(3):
            removed_any = False
            for i in range(len(best_blocks)):
                if time.time() > deadline - 1:
                    break
                trial = best_blocks[:i] + best_blocks[i+1:]
                covered = 0
                for b_idx in trial:
                    covered |= masks[b_idx]
                if covered == full_mask:
                    best_blocks = trial
                    best_len -= 1
                    removed_any = True
                    break
            if not removed_any:
                break

    if best_blocks is None or not is_valid_cover(best_blocks):
        # Fallback
        rng_fb = random.Random(999)
        best_blocks = greedy_cover(full_mask, rng_fb)
        best_blocks = reduce_cover(best_blocks)

    return [list(all_k_subsets[i]) for i in best_blocks]
