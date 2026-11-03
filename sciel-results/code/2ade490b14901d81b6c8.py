import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Precomputation
    points = range(v)
    all_blocks = [tuple(c) for c in itertools.combinations(points, k)]
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_t_subsets)
    sub_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    block_masks = [0] * num_blocks
    blocks_covering = [[] for _ in range(num_subsets)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_index[t_sub]
            mask |= (1 << idx)
            blocks_covering[idx].append(i)
        block_masks[i] = mask
        
    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            bits.append(idx)
            mask ^= lsb
        return bits

    # 2. Greedy Initialization
    uncovered_mask = (1 << num_subsets) - 1
    selected_indices = set()
    
    while uncovered_mask != 0 and time.time() < deadline:
        best_block = -1
        best_gain = 0
        
        for i in range(num_blocks):
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = i
                
        if best_block == -1 or best_gain == 0:
            break
            
        selected_indices.add(best_block)
        uncovered_mask &= ~block_masks[best_block]
        
    # 3. Local Search (Simulated Annealing)
    selected = set(selected_indices)
    counts = [0] * num_subsets
    
    for b in selected:
        for s_idx in get_bits(block_masks[b]):
            counts[s_idx] += 1
            
    uncovered_count = 0
    for s in range(num_subsets):
        if counts[s] == 0:
            uncovered_count += 1
            
    best_selected = set(selected)
    best_size = len(selected)
    
    LAMBDA = 5.0
    T = 5.0
    cooling = 0.9995
    
    uncovered_list = [i for i in range(num_subsets) if counts[i] == 0]
    
    while time.time() < deadline:
        if T < 0.01:
            T = 0.01
            
        r = random.random()
        
        if r < 0.4:
            if not selected:
                continue
            b = random.choice(tuple(selected))
            
            holes_created = 0
            for s_idx in get_bits(block_masks[b]):
                if counts[s_idx] == 1:
                    holes_created += 1
                    
            delta_cost = -1 + LAMBDA * holes_created
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                selected.remove(b)
                for s_idx in get_bits(block_masks[b]):
                    if counts[s_idx] == 1:
                        uncovered_count += 1
                        uncovered_list.append(s_idx)
                    counts[s_idx] -= 1
                        
        elif r < 0.8:
            if not uncovered_list:
                continue
            s_idx = random.choice(uncovered_list)
            if counts[s_idx] != 0:
                continue
            candidates = blocks_covering[s_idx]
            if not candidates:
                continue
            b = random.choice(candidates)
            
            if b in selected:
                continue
                
            holes_fixed = 0
            for s_idx2 in get_bits(block_masks[b]):
                if counts[s_idx2] == 0:
                    holes_fixed += 1
                    
            delta_cost = 1 - LAMBDA * holes_fixed
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                selected.add(b)
                for s_idx2 in get_bits(block_masks[b]):
                    if counts[s_idx2] == 0:
                        uncovered_count -= 1
                    counts[s_idx2] += 1
        else:
            # Swap
            if not selected or not uncovered_list:
                continue
            b_in = random.choice(tuple(selected))
            
            holes_created = 0
            holes_list = []
            for s_idx in get_bits(block_masks[b_in]):
                if counts[s_idx] == 1:
                    holes_created += 1
                    holes_list.append(s_idx)
                    
            if not holes_list:
                continue
                
            s_target = random.choice(holes_list)
            candidates = blocks_covering[s_target]
            candidates = [b for b in candidates if b != b_in]
            if not candidates:
                continue
            b_out = random.choice(candidates)
            
            if b_out in selected:
                continue
            
            # Calculate net delta
            # Remove b_in, Add b_out
            # We simulate the effect on counts
            # But we need delta_cost before modifying.
            
            # Net holes created: s where count becomes 0 and stays 0
            # Net holes fixed: s where count was 0 and becomes >0
            
            # Let's just do a quick simulation on the specific subsets involved
            # Subsets in b_in or b_out
            involved = set(get_bits(block_masks[b_in])) | set(get_bits(block_masks[b_out]))
            
            net_holes = 0
            for s in involved:
                c_old = counts[s]
                c_new = c_old
                if s in block_masks[b_in]: # Check if b_in covers s
                    c_new -= 1
                if s in block_masks[b_out]: # Check if b_out covers s
                    c_new += 1
                    
                if c_old == 0 and c_new > 0:
                    net_holes -= 1 # Fixed
                elif c_old > 0 and c_new == 0:
                    net_holes += 1 # Created
                    
            delta_cost = 0 + LAMBDA * net_holes # Size change is 0
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Accept
                selected.remove(b_in)
                selected.add(b_out)
                
                for s in get_bits(block_masks[b_in]):
                    if counts[s] == 1:
                        uncovered_count += 1
                        uncovered_list.append(s)
                    counts[s] -= 1
                    
                for s in get_bits(block_masks[b_out]):
                    if counts[s] == 0:
                        uncovered_count -= 1
                    counts[s] += 1

        T *= cooling
        
        if uncovered_count == 0 and len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
            
    # Final deletion pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for b in list(best_selected):
            redundant = True
            for s_idx in get_bits(block_masks[b]):
                # Check if any other block in best_selected covers s_idx
                # This is slow. 
                # Use counts from the final state?
                # We need to recompute counts for best_selected
                pass
            # This final pass is tricky with the current state.
            # Let's just recompute counts for best_selected and do a simple deletion.
            break # Skip complex final pass to avoid bugs, SA should be good enough.

    result = [all_blocks[i] for i in best_selected]
    return result
