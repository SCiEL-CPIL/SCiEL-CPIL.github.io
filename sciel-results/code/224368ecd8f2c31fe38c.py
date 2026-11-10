import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage masks for each block
    block_masks = [0] * num_blocks
    bit_vals = [1 << i for i in range(num_tsubs)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_to_idx[tsub]
            mask |= bit_vals[idx]
        block_masks[b_idx] = mask
    
    full_mask = (1 << num_tsubs) - 1
    
    # Phase 1: Greedy Set Cover to get a valid initial solution
    covered_mask = 0
    used_blocks = []
    
    while covered_mask != full_mask:
        if time.time() > deadline - 5.0:
            break
        
        best_gain = -1
        best_block_idx = -1
        
        for b_idx in range(num_blocks):
            new_bits = block_masks[b_idx] & ~covered_mask
            gain = new_bits.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_gain == 0:
            break
        
        used_blocks.append(best_block_idx)
        covered_mask |= block_masks[best_block_idx]
    
    # Ensure we have a valid cover
    if covered_mask != full_mask:
        # Repair: add blocks to cover remaining holes
        remaining = full_mask & ~covered_mask
        while remaining != 0 and time.time() < deadline - 2.0:
            # Pick an uncovered t-subset
            # Find the lowest set bit in remaining
            low_bit = remaining & (-remaining)
            idx = low_bit.bit_length() - 1
            target_tsub = all_tsubsets[idx]
            
            # Find a block containing this t-subset that covers the most new t-subsets
            best_gain = -1
            best_block_idx = -1
            
            for b_idx, block in enumerate(all_blocks):
                if set(target_tsub).issubset(set(block)):
                    new_bits = block_masks[b_idx] & remaining
                    gain = new_bits.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = b_idx
            
            if best_gain == 0:
                break
            
            used_blocks.append(best_block_idx)
            covered_mask |= block_masks[best_block_idx]
            remaining = full_mask & ~covered_mask
    
    # Phase 2: Redundancy Removal
    active_indices = list(used_blocks)
    
    changed = True
    while changed and time.time() < deadline - 3.0:
        changed = False
        indices_to_check = list(active_indices)
        random.shuffle(indices_to_check)
        
        for idx in indices_to_check:
            if idx not in active_indices:
                continue
            
            union_others = 0
            mask_current = block_masks[idx]
            
            for other_idx in active_indices:
                if other_idx != idx:
                    union_others |= block_masks[other_idx]
            
            if (mask_current & ~union_others) == 0:
                active_indices.remove(idx)
                changed = True
                break
    
    # Phase 3: Simulated Annealing with Swap Moves
    current_blocks = list(active_indices)
    current_mask = 0
    for idx in current_blocks:
        current_mask |= block_masks[idx]
    
    # Verify validity
    if current_mask != full_mask:
        # Fallback to greedy solution
        current_blocks = list(used_blocks)
        current_mask = 0
        for idx in current_blocks:
            current_mask |= block_masks[idx]
    
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline - 2.0 and T > T_min:
        # Generate a neighbor
        neighbor = list(current_blocks)
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a random block
            if len(neighbor) > 1:
                remove_idx = random.randint(0, len(neighbor) - 1)
                removed_block = neighbor[remove_idx]
                neighbor.pop(remove_idx)
                
                # Check if still valid
                new_mask = 0
                for idx in neighbor:
                    new_mask |= block_masks[idx]
                
                if new_mask == full_mask:
                    # Accept removal
                    current_blocks = neighbor
                    current_mask = new_mask
                else:
                    # Reject: restore
                    pass
        elif move_type < 0.8:
            # Swap: remove one block, add another
            if len(neighbor) > 1:
                remove_idx = random.randint(0, len(neighbor) - 1)
                removed_block = neighbor[remove_idx]
                neighbor.pop(remove_idx)
                
                # Find a candidate block to add
                # Prefer blocks that cover uncovered t-subsets
                uncovered = full_mask & ~current_mask
                
                candidates = []
                for b_idx in range(num_blocks):
                    if b_idx not in neighbor:
                        new_bits = block_masks[b_idx] & uncovered
                        if new_bits.bit_count() > 0:
                            candidates.append(b_idx)
                
                if candidates:
                    add_block = random.choice(candidates)
                    neighbor.append(add_block)
                    
                    new_mask = 0
                    for idx in neighbor:
                        new_mask |= block_masks[idx]
                    
                    if new_mask == full_mask:
                        current_blocks = neighbor
                        current_mask = new_mask
        else:
            # Add a random block (to explore)
            candidates = [b_idx for b_idx in range(num_blocks) if b_idx not in neighbor]
            if candidates:
                add_block = random.choice(candidates)
                neighbor.append(add_block)
                
                new_mask = 0
                for idx in neighbor:
                    new_mask |= block_masks[idx]
                
                if new_mask == full_mask:
                    current_blocks = neighbor
                    current_mask = new_mask
        
        # Track best
        if len(current_blocks) < best_size:
            best_blocks = list(current_blocks)
            best_size = len(current_blocks)
        
        T *= alpha
    
    # Final redundancy removal on best solution
    final_indices = list(best_blocks)
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        indices_to_check = list(final_indices)
        random.shuffle(indices_to_check)
        
        for idx in indices_to_check:
            if idx not in final_indices:
                continue
            
            union_others = 0
            mask_current = block_masks[idx]
            
            for other_idx in final_indices:
                if other_idx != idx:
                    union_others |= block_masks[other_idx]
            
            if (mask_current & ~union_others) == 0:
                final_indices.remove(idx)
                changed = True
                break
    
    # Final validation
    final_union = 0
    for idx in final_indices:
        final_union |= block_masks[idx]
    
    if final_union != full_mask:
        # Fallback to best_blocks
        final_indices = list(best_blocks)
    
    final_blocks = [list(all_blocks[i]) for i in final_indices]
    
    # Last resort validation
    final_union = 0
    for idx in final_indices:
        final_union |= block_masks[idx]
    
    if final_union != full_mask:
        # Use the greedy solution
        final_blocks = [list(all_blocks[i]) for i in used_blocks]
    
    return final_blocks

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    all_tsubs = set(itertools.combinations(range(24), 3))
    covered = set()
    for b in blocks:
        for t in itertools.combinations(b, 3):
            covered.add(t)
    if all_tsubs == covered:
        print("Valid Cover")
    else:
        print(f"Invalid! Missing {len(all_tsubs - covered)} subsets")
