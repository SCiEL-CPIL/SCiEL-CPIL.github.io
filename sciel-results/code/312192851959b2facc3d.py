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
        
        # Precompute gain cache for speed
        # For each block, precompute which t-subsets it covers (already in block_masks)
        
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
        
        n = len(selected_indices)
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
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy passes
    attempt = 0
    while time.time() < deadline - 5.0:  # Leave 5s for final cleanup
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
