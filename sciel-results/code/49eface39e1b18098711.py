"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    TIME_LIMIT = 55  # seconds, leave buffer
    
    # Split vertices into Small [2, 128] and Large [129, 256]
    # Any two distinct Large vertices are compatible (lcm > 256)
    # So we decide on Small set, and include all compatible Large vertices.
    
    small_vals = list(range(2, 129)) # 2..128
    large_vals = list(range(129, 257)) # 129..256
    
    n_small = len(small_vals)
    n_large = len(large_vals)
    
    # Weights
    w_small = [1.0 / v for v in small_vals]
    w_large = [1.0 / v for v in large_vals]
    total_large_weight = sum(w_large)
    
    # Conflicts in Small
    # conflict_small_mask[i] is a bitmask of small indices j that conflict with i
    conflict_small_mask = [0] * n_small
    for i in range(n_small):
        vi = small_vals[i]
        mask = 0
        for j in range(n_small):
            if i == j: continue
            vj = small_vals[j]
            if lcm(vi, vj) <= 256:
                mask |= (1 << j)
        conflict_small_mask[i] = mask
        
    # Conflicts Small -> Large
    # c_large_mask[i] is a bitmask of large indices j that conflict with small_vals[i]
    c_large_mask = [0] * n_small
    for i, v in enumerate(small_vals):
        mask = 0
        for j, u in enumerate(large_vals):
            if lcm(v, u) <= 256:
                mask |= (1 << j)
        c_large_mask[i] = mask

    # Order small vertices by weight descending (value ascending)
    # We want to pick high weight (small value) nodes first?
    # Or low value (high weight) nodes first?
    # Small values (2, 3) have high weight but high conflict penalty.
    # Large values (128) have low weight but low conflict penalty.
    # Usually, sorting by weight descending helps B&B if the bound is tight.
    # Let's sort by weight descending (index 0 is value 2, weight 0.5)
    order = sorted(range(n_small), key=lambda i: w_small[i], reverse=True)
    
    # Remap indices to order positions for faster array access?
    # Or just iterate through order.
    # Let's create arrays indexed by order position.
    # pos 0 -> small index order[0]
    # pos 1 -> small index order[1]
    
    ordered_w_small = [w_small[order[i]] for i in range(n_small)]
    ordered_conflict_small = [0] * n_small
    ordered_c_large = [0] * n_small
    
    # We need to map original small indices to order positions?
    # No, we can just keep the masks in original indices, but when we check compatibility
    # with `selected_mask`, we need to check if `selected_mask` has any bits set
    # that correspond to conflicts of `current_node`.
    
    # Actually, it's easier to work in the "order" space if we rebuild masks.
    # Let's rebuild masks in the order space.
    # pos p corresponds to original index orig_idx = order[p]
    # Conflict mask for pos p should have bits set for positions q such that
    # original index order[q] conflicts with orig_idx.
    
    # Map original index to position
    orig_to_pos = [0] * n_small
    for p, orig_idx in enumerate(order):
        orig_to_pos[orig_idx] = p
        
    for p in range(n_small):
        orig_idx = order[p]
        # Small-Small conflict
        mask = 0
        for q in range(n_small):
            if p == q: continue
            if conflict_small_mask[orig_idx] & (1 << order[q]):
                mask |= (1 << q)
        ordered_conflict_small[p] = mask
        
        # Small-Large conflict
        ordered_c_large[p] = c_large_mask[orig_idx]
        
    best_score = 0.0
    best_state_mask = 0 # Bitmask of selected small positions
    best_blocked_mask = 0 # Bitmask of blocked large indices
    
    def dfs(selected_mask, blocked_large_mask, current_score, next_pos):
        nonlocal best_score, best_state_mask, best_blocked_mask
        nonlocal start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        # Current Score is already calculated
        if current_score > best_score:
            best_score = current_score
            best_state_mask = selected_mask
            best_blocked_mask = blocked_large_mask
        
        # Upper Bound
        # Sum of weights of all remaining compatible small nodes
        # A node at pos p is compatible if (selected_mask & ordered_conflict_small[p]) == 0
        ub_add = 0.0
        for p in range(next_pos, n_small):
            if (selected_mask & ordered_conflict_small[p]) == 0:
                ub_add += ordered_w_small[p]
                
        # Tighter Bound: Subtract a lower bound on penalty
        # We must block at least the largest single conflict set among remaining compatible nodes?
        # Or just use the loose bound for speed, as bitmask ops are fast.
        if current_score + ub_add <= best_score:
            return
        
        # Branch
        for p in range(next_pos, n_small):
            # Check compatibility
            if (selected_mask & ordered_conflict_small[p]) != 0:
                continue
            
            # Include p
            new_selected_mask = selected_mask | (1 << p)
            
            # Calculate newly blocked large nodes
            new_blocked_mask = blocked_large_mask | ordered_c_large[p]
            delta_mask = new_blocked_mask & ~blocked_large_mask
            
            # Calculate penalty
            penalty = 0.0
            m = delta_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                penalty += w_large[idx]
                m ^= lsb
            
            new_score = current_score + ordered_w_small[p] - penalty
            
            dfs(new_selected_mask, new_blocked_mask, new_score, p + 1)
            
            if time.time() - start_time > TIME_LIMIT:
                return
            
            # Exclude p
            # This is implicit by just moving to next p in the loop?
            # No, standard B&B branches: Include or Exclude.
            # If we just iterate p, we are doing "Include p, then try next p without p".
            # This is equivalent to branching.
            # However, we need to ensure we explore the "Exclude p" branch.
            # The loop structure `for p in range...` where we recurse on `p+1` after including `p`
            # effectively explores the branch where `p` is included.
            # The branch where `p` is excluded is explored when the loop increments to `p+1`
            # WITHOUT having included `p` in `selected_mask`.
            # Wait, the current logic:
            # 1. Check if p compatible.
            # 2. Recurse with p included (next_pos = p+1).
            # 3. Loop continues to p+1. The state `selected_mask` is unchanged.
            #    So the next iteration considers p+1 with the original `selected_mask`.
            #    This IS the "Exclude p" branch.
            # So the loop handles both branches correctly.
            
    # Initial call
    dfs(0, 0, 0.0, 0)
    
    # Reconstruct result
    if best_state_mask == 0 and best_blocked_mask == 0:
        # Fallback: pick a single node with max weight (smallest value)
        # Actually, if best_score is 0, it means we didn't find anything?
        # But we start with score 0.
        # If we pick no small nodes, score is total_large_weight.
        # So best_score should be at least total_large_weight.
        pass
    
    result_small = []
    for p in range(n_small):
        if best_state_mask & (1 << p):
            result_small.append(small_vals[order[p]])
            
    result_large = []
    for j in range(n_large):
        if not (best_blocked_mask & (1 << j)):
            result_large.append(large_vals[j])
            
    return sorted(result_small + result_large)
# EVOLVE-BLOCK-END
