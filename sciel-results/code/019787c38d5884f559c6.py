import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Map t-subsets to IDs
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute k-subsets (blocks) and their covered t-subset bitsets
    # Using Python integers as bitsets for fast intersection and counting
    all_k_subsets = list(itertools.combinations(range(v), k))
    block_bitsets = []
    for block in all_k_subsets:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            mask |= (1 << t_subset_to_id[t_sub])
        block_bitsets.append(mask)
        
    full_mask = (1 << num_t_subsets) - 1
    
    # 3. Greedy with Random Restarts + Reduction
    best_blocks = None
    best_count = float('inf')
    
    def reduce_blocks(blocks_indices):
        """Try to remove redundant blocks from the selected set."""
        selected_indices = set(blocks_indices)
        # We need to know if a block is redundant.
        # A block is redundant if all t-subsets it covers are also covered by other selected blocks.
        # To do this efficiently, we can maintain a count of how many selected blocks cover each t-subset.
        # However, rebuilding the count from scratch for each removal attempt is O(N_blocks * C(k,t)).
        # Given N_blocks ~ 150 and C(7,4)=35, this is 150*35 = 5250 operations per check.
        # Checking all 150 blocks is 150*5250 = ~787k operations. This is fast enough.
        
        # Let's try a simpler approach: 
        # Maintain a counter for each t-subset.
        # If we remove a block, decrement counts. If any count hits 0, it's not redundant.
        
        # We will iterate through the selected blocks and try to remove them.
        # We need a way to quickly check redundancy.
        
        # Let's use the bitsets.
        # For a candidate block B, it is redundant if 
        # (OR of bitsets of all other selected blocks) covers (bitset of B).
        # i.e., (other_union | B_bitset) == other_union  =>  B_bitset is subset of other_union.
        
        # Calculating other_union for each candidate is O(N_blocks) bit operations.
        # Total O(N_blocks^2) bit operations. 150^2 = 22500 big int ops. Very fast.
        
        current_selected = list(blocks_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(current_selected)):
                if time.time() > deadline:
                    return current_selected
                
                idx_to_check = current_selected[i]
                b_mask = block_bitsets[idx_to_check]
                
                # Calculate union of all other blocks
                other_union = 0
                for j in range(len(current_selected)):
                    if i == j:
                        continue
                    other_union |= block_bitsets[current_selected[j]]
                
                # Check if b_mask is subset of other_union
                if (b_mask & other_union) == b_mask:
                    # Redundant
                    current_selected.pop(i)
                    changed = True
                    break # Restart loop since indices shifted
                    
        return current_selected

    while time.time() < deadline:
        uncovered = full_mask
        current_indices = []
        
        while uncovered:
            max_gain = -1
            best_candidates = []
            
            # Find block with max gain
            for i, mask in enumerate(block_bitsets):
                # Intersection size using bit_count
                gain = (uncovered & mask).bit_count()
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [i]
                elif gain == max_gain and gain > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
                
            # Pick a candidate
            chosen_idx = random.choice(best_candidates)
            current_indices.append(chosen_idx)
            
            # Update uncovered
            uncovered &= ~block_bitsets[chosen_idx]
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
        
        # Reduction phase
        reduced_indices = reduce_blocks(current_indices)
        
        if len(reduced_indices) < best_count:
            best_count = len(reduced_indices)
            best_blocks = [list(all_k_subsets[i]) for i in reduced_indices]
            
            if best_count <= 131:
                break

    if best_blocks is None:
        # Fallback: run a deterministic greedy without reduction to ensure valid output
        # This is a safety net, should not be reached if first run completes
        uncovered = full_mask
        current_indices = []
        while uncovered:
            max_gain = -1
            chosen_idx = -1
            for i, mask in enumerate(block_bitsets):
                gain = (uncovered & mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    chosen_idx = i
            if chosen_idx == -1: break
            current_indices.append(chosen_idx)
            uncovered &= ~block_bitsets[chosen_idx]
        best_blocks = [list(all_k_subsets[i]) for i in current_indices]

    return best_blocks
