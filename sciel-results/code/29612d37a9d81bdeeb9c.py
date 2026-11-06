import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    """
    Find a covering design C(v,k,t): choose as few k-subsets (blocks) of {0,...,v-1}
    as possible such that every t-subset is contained in at least one block.
    
    Strategy:
    1. Greedy set cover with bitmask acceleration to get an initial solution.
    2. Local search (add/remove/swap moves) to reduce the number of blocks.
    3. Simulated-annealing-like restarts within time budget.
    """
    start_time = time.time()
    DEADLINE = start_time + 130  # Leave 20s margin for safety
    
    # Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_count = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Precompute all k-subsets and their coverage masks
    all_k_subs = list(itertools.combinations(range(v), k))
    k_sub_count = len(all_k_subs)
    
    # Build coverage masks: bit i set if block covers t-subset i
    block_masks = []
    for block in all_k_subs:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Also precompute: for each t-subset index, which blocks cover it
    # This is useful for redundancy checks
    # But with 3876 t-subsets and 50388 blocks, this could be large.
    # Instead, we'll use bitmasks for coverage tracking.
    
    full_mask = (1 << t_sub_count) - 1
    
    def greedy_set_cover():
        """Standard greedy: repeatedly pick block covering most uncovered t-subsets."""
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(k_sub_count):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        return selected
    
    def is_valid_cover(selected_indices):
        """Check if selected blocks cover all t-subsets."""
        covered = 0
        for idx in selected_indices:
            covered |= block_masks[idx]
        return covered == full_mask
    
    def count_redundant(selected_indices):
        """Count how many blocks are redundant (removing them still leaves a valid cover)."""
        selected_set = set(selected_indices)
        redundant_count = 0
        for idx in selected_indices:
            # Check if removing idx still covers everything
            covered_without = 0
            for j in selected_indices:
                if j != idx:
                    covered_without |= block_masks[j]
            if covered_without == full_mask:
                redundant_count += 1
        return redundant_count
    
    def remove_redundant_blocks(selected_indices):
        """Remove redundant blocks until none remain. Returns new list."""
        selected_set = set(selected_indices)
        changed = True
        while changed:
            changed = False
            new_selected = []
            for idx in selected_indices:
                # Check if this block is needed
                covered_without = 0
                for j in selected_indices:
                    if j != idx:
                        covered_without |= block_masks[j]
                if covered_without == full_mask:
                    changed = True
                else:
                    new_selected.append(idx)
            selected_indices = new_selected
        return selected_indices
    
    def local_search(selected_indices, max_iterations=5000):
        """
        Local search to reduce block count.
        Moves:
        1. Remove a redundant block (if any).
        2. Swap: remove block A, add block B (if A is redundant after adding B, or if B covers what A uniquely covers).
        3. Try to replace two blocks with one (if possible).
        """
        selected_set = set(selected_indices)
        current_count = len(selected_indices)
        
        # First, remove all redundant blocks
        selected_indices = remove_redundant_blocks(selected_indices)
        current_count = len(selected_indices)
        
        for iteration in range(max_iterations):
            if time.time() > DEADLINE:
                break
            
            # Compute coverage for each t-subset: how many selected blocks cover it
            # For efficiency, track per-t-subset count
            t_sub_counts = [0] * t_sub_count
            for idx in selected_indices:
                mask = block_masks[idx]
                # Extract set bits
                m = mask
                while m:
                    low_bit = m & (-m)
                    bit_pos = low_bit.bit_length() - 1
                    t_sub_counts[bit_pos] += 1
                    m ^= low_bit
            
            # Find blocks that are redundant (all their t-subsets are covered by >= 2 blocks)
            redundant_candidates = []
            for idx in selected_indices:
                mask = block_masks[idx]
                m = mask
                is_redundant = True
                while m:
                    low_bit = m & (-m)
                    bit_pos = low_bit.bit_length() - 1
                    if t_sub_counts[bit_pos] <= 1:
                        is_redundant = False
                        break
                    m ^= low_bit
                if is_redundant:
                    redundant_candidates.append(idx)
            
            if redundant_candidates:
                # Remove one redundant block
                idx_to_remove = random.choice(redundant_candidates)
                selected_indices.remove(idx_to_remove)
                selected_set.discard(idx_to_remove)
                current_count -= 1
                continue
            
            # Try swap: for each selected block, try replacing it with an unselected block
            # that results in a valid cover with same or fewer blocks
            # This is expensive, so do it probabilistically
            if random.random() < 0.3:
                # Pick a random selected block to potentially replace
                if selected_indices:
                    idx_to_replace = random.choice(selected_indices)
                    # Find blocks that cover at least the unique t-subsets of idx_to_replace
                    # Unique t-subsets: those covered only by idx_to_replace
                    unique_mask = 0
                    mask = block_masks[idx_to_replace]
                    m = mask
                    while m:
                        low_bit = m & (-m)
                        bit_pos = low_bit.bit_length() - 1
                        if t_sub_counts[bit_pos] == 1:
                            unique_mask |= (1 << bit_pos)
                        m ^= low_bit
                    
                    if unique_mask == 0:
                        # No unique coverage, can remove and add any block that maintains coverage
                        # Actually if no unique, it's redundant, should have been caught above
                        pass
                    else:
                        # Find blocks that cover unique_mask
                        best_new_block = -1
                        best_extra = -1
                        for j in range(k_sub_count):
                            if j in selected_set:
                                continue
                            if (block_masks[j] & unique_mask) != unique_mask:
                                continue
                            # Check if adding j and removing idx_to_replace still covers all
                            # Quick check: does j cover enough of what others don't?
                            # For now, just accept if it covers unique_mask
                            # Count how many additional t-subsets it covers (beyond unique)
                            extra = (block_masks[j] & ~block_masks[idx_to_replace]).bit_count()
                            if extra > best_extra:
                                best_extra = extra
                                best_new_block = j
                        
                        if best_new_block != -1:
                            selected_indices.remove(idx_to_replace)
                            selected_set.discard(idx_to_replace)
                            selected_indices.append(best_new_block)
                            selected_set.add(best_new_block)
                            # Count stays the same, but structure may improve for future removals
        
        return selected_indices
    
    # Phase 1: Greedy initial solution
    initial_solution = greedy_set_cover()
    best_solution = list(initial_solution)
    best_count = len(best_solution)
    
    # Phase 2: Local search
    current_solution = list(initial_solution)
    current_solution = local_search(current_solution, max_iterations=10000)
    if len(current_solution) < best_count:
        best_solution = current_solution
        best_count = len(current_solution)
    
    # Phase 3: Multiple restarts with perturbation
    while time.time() < DEADLINE - 10:
        # Perturb the current best solution
        perturbed = list(best_solution)
        # Remove 2-5 random blocks
        num_remove = random.randint(2, min(5, len(perturbed)))
        for _ in range(num_remove):
            if perturbed:
                idx = random.choice(perturbed)
                perturbed.remove(idx)
        
        # Try to repair: add blocks to make it valid again
        # Use greedy to add blocks
        covered = 0
        for idx in perturbed:
            covered |= block_masks[idx]
        
        uncovered = full_mask & ~covered
        selected_set = set(perturbed)
        
        while uncovered and time.time() < DEADLINE - 5:
            best_idx = -1
            best_gain = 0
            for i in range(k_sub_count):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            perturbed.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        # Check validity
        if is_valid_cover(perturbed):
            # Local search on the repaired solution
            repaired = local_search(perturbed, max_iterations=5000)
            if len(repaired) < best_count:
                best_solution = repaired
                best_count = len(repaired)
    
    # Final cleanup: remove any remaining redundant blocks
    best_solution = remove_redundant_blocks(best_solution)
    
    # Convert indices to actual blocks
    result = [list(all_k_subs[idx]) for idx in best_solution]
    
    # Verify validity
    if not is_valid_cover(best_solution):
        # Fallback: return greedy solution
        result = [list(all_k_subs[idx]) for idx in initial_solution]
    
    return result
