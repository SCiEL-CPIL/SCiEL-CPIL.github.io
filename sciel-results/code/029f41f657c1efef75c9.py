import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # 2. Precompute all k-blocks and their coverage masks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    block_masks = [0] * num_k
    for idx, block in enumerate(k_blocks):
        mask = 0
        for combo in itertools.combinations(block, t):
            mask |= (1 << t_index_map[combo])
        block_masks[idx] = mask
    
    full_mask = (1 << num_t) - 1
    
    def _greedy_pass(seed):
        """One greedy set cover pass with optional randomization."""
        if time.time() > deadline:
            return None
        
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            # Find block with max coverage of uncovered subsets
            best_idx = -1
            best_count = -1
            
            # Randomize tie-breaking for diversity
            indices = list(range(num_k))
            if seed % 2 == 0:
                rng.shuffle(indices)
            
            for i in indices:
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        return selected
    
    def _remove_redundant(selected_indices):
        """Remove redundant blocks from a valid covering."""
        if not selected_indices:
            return selected_indices
        
        kept = list(selected_indices)
        changed = True
        
        while changed and time.time() < deadline:
            changed = False
            new_kept = []
            for i in range(len(kept)):
                idx = kept[i]
                # Check if this block's coverage is fully covered by others
                others_mask = 0
                for j in range(len(kept)):
                    if j != i:
                        others_mask |= block_masks[kept[j]]
                
                # If all t-subsets covered by this block are also covered by others
                if (block_masks[idx] & ~others_mask) == 0:
                    changed = True
                else:
                    new_kept.append(idx)
            
            if changed:
                kept = new_kept
        
        return kept
    
    def _walksat_minimize(selected_indices, max_steps=5000):
        """WalkSAT-style local search to minimize number of blocks while maintaining coverage.
        
        State: set of selected block indices.
        Objective: minimize |selected| subject to union of masks == full_mask.
        Moves:
          1. Remove a redundant block (improvement)
          2. Add a block (may enable future removals)
          3. Swap: remove one, add one (if net size decreases or coverage maintained)
        """
        if time.time() > deadline:
            return selected_indices
        
        rng = random.Random()
        current = set(selected_indices)
        best = set(current)
        best_len = len(current)
        
        # Precompute: for each block, which other blocks can cover its unique t-subsets
        # This is expensive, so we'll use a simpler approach
        
        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            
            # Compute coverage mask of current set
            cov = 0
            for idx in current:
                cov |= block_masks[idx]
            
            # If not fully covered, need to add blocks
            if cov != full_mask:
                uncovered = full_mask & ~cov
                # Find block with max gain on uncovered
                best_gain = -1
                best_add = -1
                for i in range(num_k):
                    if i in current:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_add = i
                if best_add != -1 and best_gain > 0:
                    current.add(best_add)
                continue
            
            # Fully covered: try to remove redundant blocks
            # Find a block whose removal doesn't break coverage
            removable = []
            for idx in list(current):
                others_cov = 0
                for j in current:
                    if j != idx:
                        others_cov |= block_masks[j]
                if (block_masks[idx] & ~others_cov) == 0:
                    removable.append(idx)
            
            if removable:
                # Remove one (prefer randomly for diversity)
                idx_to_remove = rng.choice(removable)
                current.discard(idx_to_remove)
                if len(current) < best_len:
                    best = set(current)
                    best_len = len(current)
                continue
            
            # No single removal helps. Try swap: remove one, add one
            # For each candidate to remove, check if we can find a block that covers its unique parts
            swap_done = False
            for idx_to_remove in list(current):
                if time.time() > deadline - 2.0:
                    break
                # What does this block uniquely cover?
                others_cov = 0
                for j in current:
                    if j != idx_to_remove:
                        others_cov |= block_masks[j]
                unique_parts = block_masks[idx_to_remove] & ~others_cov
                
                if unique_parts == 0:
                    # Actually redundant, remove it
                    current.discard(idx_to_remove)
                    if len(current) < best_len:
                        best = set(current)
                        best_len = len(current)
                    swap_done = True
                    break
                
                # Find a block not in current that covers all unique_parts
                # and doesn't add too many extra blocks (we'll remove the old one)
                best_add = -1
                best_extra = float('inf')
                for i in range(num_k):
                    if i in current:
                        continue
                    # Does it cover all unique parts?
                    if (unique_parts & ~block_masks[i]) != 0:
                        continue
                    # Extra coverage it adds (not in current coverage minus removed block)
                    new_cov = (others_cov | block_masks[i])
                    extra = new_cov.bit_count() - others_cov.bit_count()
                    if extra < best_extra:
                        best_extra = extra
                        best_add = i
                
                if best_add != -1:
                    # Perform swap
                    current.discard(idx_to_remove)
                    current.add(best_add)
                    # Size unchanged, but structure changed - may enable future removals
                    swap_done = True
                    break
            
            if not swap_done:
                # Stuck: try adding a random block to escape local minimum
                if len(current) < num_k:
                    candidates = [i for i in range(num_k) if i not in current]
                    if candidates:
                        add_idx = rng.choice(candidates)
                        current.add(add_idx)
        
        return best
    
    def _polish(selected_indices):
        """Final polishing: repeated redundancy removal and local swaps."""
        if time.time() > deadline:
            return selected_indices
        
        result = _remove_redundant(selected_indices)
        
        # Try a few rounds of WalkSAT-style minimization
        for _ in range(3):
            if time.time() > deadline - 1.0:
                break
            result = _walksat_minimize(result, max_steps=2000)
            result = _remove_redundant(result)
        
        return result
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Run multiple randomized greedy passes
    attempt = 0
    while time.time() < deadline - 20.0:  # Leave 20s for polishing
        selected = _greedy_pass(attempt)
        if selected is None:
            break
        
        # Remove redundant blocks
        selected = _remove_redundant(selected)
        
        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        
        # Early stop if we've found a very good solution
        if best_len <= 340:
            break
        
        attempt += 1
    
    # Phase 2: Polish the best solution found
    if best_blocks is not None and time.time() < deadline - 1.0:
        best_indices = [i for i in range(num_k) if k_blocks[i] in best_blocks]
        # Reconstruct indices properly
        block_to_idx = {tuple(b): i for i, b in enumerate(k_blocks)}
        best_indices = [block_to_idx[tuple(b)] for b in best_blocks]
        
        polished = _polish(best_indices)
        if len(polished) < best_len:
            best_len = len(polished)
            best_blocks = [list(k_blocks[i]) for i in polished]
    
    # Fallback if no solution found
    if best_blocks is None:
        # Deterministic greedy
        uncovered = full_mask
        selected = []
        selected_set = set()
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            for i in range(num_k):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        
        if selected:
            selected = _remove_redundant(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            # Last resort: return empty (invalid but avoids crash)
            best_blocks = []
    
    return best_blocks
