import itertools
import time
import random
import sys
import math

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_masks, deadline):
    """Remove redundant blocks from selected set using bitsets."""
    if not selected:
        return selected
    
    sol = list(selected)
    
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in range(len(sol)):
            b = sol[i]
            M_b = block_masks[b]
            
            # Compute union of all other blocks
            union_others = 0
            for j in range(len(sol)):
                if j != i:
                    union_others |= block_masks[sol[j]]
            
            # Check if M_b is subset of union_others
            if (M_b & ~union_others) == 0:
                sol.pop(i)
                changed = True
                break
    
    return sol

def _complete_cover(selected, block_masks, sub_to_blocks, n_t, deadline):
    """Complete a partial cover to a full valid cover using greedy."""
    sol = list(selected)
    sol_set = set(sol)
    
    # Calculate current union
    union = 0
    for b in sol:
        union |= block_masks[b]
        
    uncovered = (1 << n_t) - 1
    uncovered &= ~union
    
    while uncovered and _deadline_ok(deadline):
        # Pick a random uncovered subset
        # For efficiency in completion, just pick lowest or random
        # Lowest is deterministic and fast
        lowest_bit = uncovered & -uncovered
        s_idx = lowest_bit.bit_length() - 1
        
        candidates = sub_to_blocks[s_idx]
        
        # Find best candidate
        best_block = -1
        best_gain = 0
        for b in candidates:
            if b in sol_set:
                continue
            gain = (uncovered & block_masks[b]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b
        
        if best_block == -1:
            # Should not happen if cover exists
            break
            
        sol.append(best_block)
        sol_set.add(best_block)
        uncovered &= ~block_masks[best_block]
        
    return sol

def _simulated_annealing(selected, block_masks, sub_to_blocks, n_t, deadline):
    """
    Perform Simulated Annealing on the set of blocks.
    Objective: Minimize |S| + Penalty * Uncovered
    """
    sol = list(selected)
    sol_set = set(sol)
    
    # Initial state evaluation
    union = 0
    for b in sol:
        union |= block_masks[b]
    uncovered_mask = ((1 << n_t) - 1) & ~union
    n_uncovered = uncovered_mask.bit_count()
    
    # Penalty should be larger than max possible reduction in |S|
    # Max |S| is ~N. So Penalty = N + 1 is safe.
    penalty = len(block_masks) + 1 
    current_cost = len(sol) + penalty * n_uncovered
    
    best_sol = list(sol)
    best_cost = current_cost
    
    # SA Parameters
    T_start = 50.0
    T_end = 1.0
    cooling_rate = 0.95
    T = T_start
    
    # Estimate max steps
    # Each step is O(N) for union update if we recompute, or O(1) if we maintain.
    # Maintaining union for add/remove is O(1) for mask, but checking uncovered count is O(1) if we maintain mask.
    # Let's maintain uncovered_mask.
    
    # We need to be careful with time.
    # Each iteration:
    # 1. Generate move
    # 2. Update state (add/remove block)
    # 3. Calculate delta cost
    # 4. Accept/Reject
    # 5. Update T
    
    # Move types:
    # - Add random block
    # - Remove random block
    # - Swap random block with random other
    
    n_blocks = len(block_masks)
    
    # Precompute random block list? No, just randint.
    
    while T > T_end and _deadline_ok(deadline):
        T *= cooling_rate
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a random block
            if not sol:
                continue
            idx = random.randint(0, len(sol) - 1)
            b_remove = sol[idx]
            M_remove = block_masks[b_remove]
            
            # Check if removal is possible (always possible, just increases cost)
            # New uncovered = uncovered_mask | M_remove
            # But we must ensure we don't remove a block that is the ONLY one covering some subset?
            # No, SA allows invalid states.
            
            new_uncovered_mask = uncovered_mask | M_remove
            new_n_uncovered = new_uncovered_mask.bit_count()
            new_cost = (len(sol) - 1) + penalty * new_n_uncovered
            
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                sol.pop(idx)
                sol_set.remove(b_remove)
                uncovered_mask = new_uncovered_mask
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sol = list(sol)
                    
        elif move_type < 0.8:
            # Add a random block
            b_add = random.randint(0, n_blocks - 1)
            if b_add in sol_set:
                continue
            M_add = block_masks[b_add]
            
            new_uncovered_mask = uncovered_mask & ~M_add
            new_n_uncovered = new_uncovered_mask.bit_count()
            new_cost = (len(sol) + 1) + penalty * new_n_uncovered
            
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                sol.append(b_add)
                sol_set.add(b_add)
                uncovered_mask = new_uncovered_mask
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sol = list(sol)
                    
        else:
            # Swap
            if len(sol) < 1:
                continue
            idx = random.randint(0, len(sol) - 1)
            b_remove = sol[idx]
            
            b_add = random.randint(0, n_blocks - 1)
            if b_add in sol_set and b_add != b_remove:
                continue
                
            M_remove = block_masks[b_remove]
            M_add = block_masks[b_add]
            
            # Remove b_remove, Add b_add
            # New uncovered = (uncovered_mask | M_remove) & ~M_add
            new_uncovered_mask = (uncovered_mask | M_remove) & ~M_add
            new_n_uncovered = new_uncovered_mask.bit_count()
            
            # Size stays same
            new_cost = len(sol) + penalty * new_n_uncovered
            
            delta = new_cost - current_cost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                sol[idx] = b_add
                sol_set.remove(b_remove)
                sol_set.add(b_add)
                uncovered_mask = new_uncovered_mask
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sol = list(sol)
    
    return best_sol

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their bitmasks
    blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            mask |= (1 << idx)
        blocks.append(block)
        block_masks.append(mask)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            sub_to_blocks[idx].append(i)
            m ^= lb
            
    if not _deadline_ok(deadline):
        return []

    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 10
    top_k = 5
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        uncovered = (1 << n_t) - 1
        selected = []
        selected_set = set()
        
        while uncovered:
            if not _deadline_ok(deadline):
                break
            
            lowest_bit = uncovered & -uncovered
            s_idx = lowest_bit.bit_length() - 1
            
            candidates = sub_to_blocks[s_idx]
            
            # Calculate gains
            gains = []
            for b in candidates:
                gain = (uncovered & block_masks[b]).bit_count()
                gains.append((gain, b))
            
            # Sort by gain descending
            gains.sort(reverse=True)
            
            # Pick randomly among top candidates
            if restart == 0:
                best_block = gains[0][1]
            else:
                k = min(top_k, len(gains))
                best_block = random.choice(gains[:k])[1]
            
            selected.append(best_block)
            selected_set.add(best_block)
            
            uncovered &= ~block_masks[best_block]
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_masks, deadline)
        
        # Simulated Annealing
        if _deadline_ok(deadline):
            sa_selected = _simulated_annealing(selected, block_masks, sub_to_blocks, n_t, deadline)
            
            # Complete if SA left it invalid
            sa_selected = _complete_cover(sa_selected, block_masks, sub_to_blocks, n_t, deadline)
            
            # Prune again
            sa_selected = _prune_redundant(sa_selected, block_masks, deadline)
            
            if len(sa_selected) < best_size:
                best_size = len(sa_selected)
                best_selected = list(sa_selected)
                
        # Also keep the greedy one if it's better
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
    
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_masks, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
