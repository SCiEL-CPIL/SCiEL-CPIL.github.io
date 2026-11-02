import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitmask for each block
    block_masks = [0] * num_blocks
    # t_to_blocks[j] = list of block indices that cover t-subset j
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            t_to_blocks[idx].append(b_idx)
        block_masks[b_idx] = mask

    best_blocks = None
    best_count = float('inf')
    
    # Helper: Extract set bits from mask
    def extract_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        return bits

    # Helper: Prune redundant blocks
    # Returns list of remaining block indices
    def prune_redundant(selected_indices):
        if not selected_indices:
            return []
        
        # Count coverage for each t-subset
        cover_counts = [0] * num_t_subsets
        for b_idx in selected_indices:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                m ^= lsb
        
        improved = True
        current_blocks = list(selected_indices)
        
        while improved and time.time() < deadline:
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                # Check if redundant: all t-subsets in this block have count > 1
                redundant = True
                m = block_masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    # Remove it
                    m = block_masks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
                    improved = True
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
            
        return current_blocks

    # Helper: Local Search with Swaps
    def local_search_improve(block_indices, max_time=5.0):
        current_blocks = list(block_indices)
        selected_set = set(current_blocks)
        
        # Precompute unique t-subsets for each selected block? 
        # This changes after swaps. We'll compute on the fly or maintain counts.
        
        # To make swaps efficient, we need to know if a new block covers the "unique" parts of the old block.
        # Let's maintain cover_counts.
        cover_counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += 1
                m ^= lsb
        
        improved = True
        last_change_time = time.time()
        
        while improved and time.time() < deadline and (time.time() - last_change_time) < max_time:
            improved = False
            # Shuffle to find different swap opportunities
            random.shuffle(current_blocks)
            
            for b_old in current_blocks[:]:
                if time.time() > deadline:
                    return current_blocks
                
                # Find unique t-subsets covered by b_old
                unique_mask = 0
                m = block_masks[b_old]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        unique_mask |= lsb
                    m ^= lsb
                
                if unique_mask == 0:
                    # This block is redundant, remove it
                    m = block_masks[b_old]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
                    current_blocks.remove(b_old)
                    selected_set.remove(b_old)
                    improved = True
                    last_change_time = time.time()
                    continue
                
                # Try to find a block b_new that covers unique_mask
                # Optimization: Only try blocks that cover the "rarest" t-subset in unique_mask?
                # Or just iterate a sample of unselected blocks?
                # Iterating all 75k blocks for every b_old is too slow (75k * 200 = 15M checks per pass).
                # Let's pick a random sample of unselected blocks, or blocks that cover one of the unique t-subsets.
                
                # Get one unique t-subset to guide the search
                # Pick the first set bit in unique_mask
                lsb = unique_mask & -unique_mask
                target_t_idx = lsb.bit_length() - 1
                
                # Candidate new blocks must cover target_t_idx
                candidates = t_to_blocks[target_t_idx]
                
                # Shuffle candidates to avoid bias
                random.shuffle(candidates)
                
                found_swap = False
                for b_new in candidates:
                    if b_new in selected_set:
                        continue
                    
                    # Check if b_new covers ALL of unique_mask
                    if (block_masks[b_new] & unique_mask) == unique_mask:
                        # Valid swap!
                        # Perform swap
                        # Remove b_old
                        m = block_masks[b_old]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] -= 1
                            m ^= lsb
                        
                        # Add b_new
                        m = block_masks[b_new]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_counts[idx] += 1
                            m ^= lsb
                        
                        current_blocks.remove(b_old)
                        current_blocks.append(b_new)
                        selected_set.remove(b_old)
                        selected_set.add(b_new)
                        
                        improved = True
                        last_change_time = time.time()
                        found_swap = True
                        break
                
                # If no swap found for this b_old, continue to next
                # If swap found, we restart the loop or continue? 
                # Continuing is fine, but counts changed.
        
        return current_blocks

    # Main Loop: Multiple Greedy Instances
    max_runs = 100 # Adjust based on speed
    
    for run_idx in range(max_runs):
        if time.time() > deadline:
            break
            
        random.seed(run_idx)
        
        # --- Phase 1: Fast Greedy ---
        uncovered_mask = (1 << num_t_subsets) - 1
        selected_indices = []
        selected_set = set()
        
        # Initial gain: C(k, t)
        init_gain = len(list(itertools.combinations(range(k), t)))
        gain = [init_gain] * num_blocks
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            # Find max gain
            max_gain = -1
            best_candidates = []
            
            # Linear scan
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                g = gain[b_idx]
                # Add small noise for randomization
                g += random.random() * 0.01
                if g > max_gain:
                    max_gain = g
                    best_candidates = [b_idx]
                elif abs(g - max_gain) < 1e-9:
                    best_candidates.append(b_idx)
            
            if not best_candidates:
                break
                
            chosen_idx = random.choice(best_candidates)
            selected_indices.append(chosen_idx)
            selected_set.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered_mask & block_masks[chosen_idx]
            uncovered_mask &= ~block_masks[chosen_idx]
            
            # Update gains for affected blocks
            m = newly_covered
            while m:
                lsb = m & -m
                t_idx = lsb.bit_length() - 1
                m ^= lsb
                for b_prime in t_to_blocks[t_idx]:
                    if b_prime not in selected_set:
                        gain[b_prime] -= 1
                        if gain[b_prime] < 0:
                            gain[b_prime] = 0
        
        if uncovered_mask != 0:
            continue
            
        # --- Phase 2: Prune ---
        selected_indices = prune_redundant(selected_indices)
        
        # --- Phase 3: Local Search (Swaps) ---
        # Only do local search if we have time
        if time.time() < deadline - 10.0:
            selected_indices = local_search_improve(selected_indices)
            
        if len(selected_indices) < best_count:
            best_count = len(selected_indices)
            best_blocks = [list(all_blocks[i]) for i in selected_indices]

    if best_blocks is None:
        return []
        
    return best_blocks
