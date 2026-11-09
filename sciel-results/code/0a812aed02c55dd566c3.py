import time
import math
import random

def construct():
    start_time = time.time()
    # Use 85s out of 90s to be safe
    deadline = start_time + 85.0
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 1
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    nv = len(valid_nodes)
    
    # Build conflict graph using bitsets
    # conflicts[i] is a bitmask of valid nodes that conflict with valid_nodes[i]
    conflicts = [0] * nv
    for i_idx in range(nv):
        v = valid_nodes[i_idx]
        for i_j in range(i_idx + 1, nv):
            u = valid_nodes[i_j]
            if v + u in squares:
                conflicts[i_idx] |= (1 << i_j)
                conflicts[i_j] |= (1 << i_idx)
                
    def popcount(x):
        return bin(x).count('1')
        
    def mask_to_set(mask):
        s = set()
        for i in range(nv):
            if mask & (1 << i):
                s.add(valid_nodes[i])
        return s

    # Greedy MIS
    def greedy_mis(order_indices):
        chosen_mask = 0
        for idx in order_indices:
            if (conflicts[idx] & chosen_mask) == 0:
                chosen_mask |= (1 << idx)
        return chosen_mask

    # Local Search
    # Returns best mask found
    def local_search(mask, max_iters=2000):
        mask = mask
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # 1. Try adding single nodes
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
            
            # 2. Try swapping 1 for 1 (size same, might help)
            # For each i not in mask, try to remove one conflict j in mask
            for i in range(nv):
                if mask & (1 << i):
                    continue
                conf_in_mask = conflicts[i] & mask
                if conf_in_mask == 0:
                    # Can add directly, should have been caught above
                    mask |= (1 << i)
                    improved = True
                    continue
                # Try removing each bit in conf_in_mask
                temp = conf_in_mask
                while temp:
                    r_bit = temp & (-temp)
                    r_idx = r_bit.bit_length() - 1
                    # Check if i conflicts with any other node in mask besides r_idx
                    if (conflicts[i] & (mask ^ r_bit)) == 0:
                        mask = (mask & ~r_bit) | (1 << i)
                        improved = True
                        break
                    temp ^= r_bit

            # 3. Try swapping 2 for 1 (size increases by 1)
            # For each pair (i, j) not in mask, try to remove one node k in mask
            # This is O(nv^2 * nv) which is too slow if done naively.
            # Optimization: 
            # For each i not in mask, find conflicts in mask.
            # If we can remove one conflict k to allow i, and then add j?
            # Actually, just try to add i and j simultaneously.
            # Condition: (conflicts[i] | conflicts[j]) & (mask ^ k_bit) == 0 for some k in mask.
            # And i, j must not conflict with each other.
            
            # Let's iterate i, j
            # To speed up, only consider i, j that are "close" to being addable?
            # Or just limit the scope.
            
            # A simpler 2-swap: 
            # Pick i not in mask.
            # Let C_i = conflicts[i] & mask.
            # If we remove ALL of C_i, we can add i.
            # If |C_i| == 1, we did it in step 2.
            # If |C_i| == 2, say {k1, k2}, we remove k1, k2 and add i, j?
            # No, we want to increase size.
            # Remove 1, Add 2.
            
            # Strategy:
            # For each i not in mask:
            #   C_i = conflicts[i] & mask
            #   If |C_i| <= 1: handled.
            #   If |C_i| == 2:
            #      Let k1, k2 be the conflicts.
            #      If we remove k1, can we add i? No, k2 still conflicts.
            #      We need to remove k1 AND k2 to add i.
            #      That's Remove 2, Add 1. Size decreases.
            #      
            # To get Remove 1, Add 2:
            # We need to find i, j not in mask such that:
            # 1. i, j do not conflict with each other.
            # 2. The set of conflicts of i and j in mask is a subset of {k} for some k in mask.
            #    i.e., (conflicts[i] | conflicts[j]) & mask is a subset of {k}.
            #    This means all conflicts of i and j in mask are exactly k (or subset).
            #    If we remove k, both i and j can be added.
            
            # Implementation:
            # Precompute for each i not in mask: C_i = conflicts[i] & mask
            # We want pairs (i, j) such that C_i | C_j is a singleton {k}.
            # This implies C_i is subset of {k} and C_j is subset of {k}.
            # So C_i and C_j must be 0 or {k}.
            
            # Group nodes by their conflict set in mask?
            # Since mask changes, this is hard to precompute.
            
            # Alternative:
            # Iterate k in mask.
            # Let S_k = { i not in mask | (conflicts[i] & mask) == (1 << k) or (conflicts[i] & mask) == 0 }
            # Actually, if (conflicts[i] & mask) == 0, we add i in step 1.
            # So we look for i with (conflicts[i] & mask) == (1 << k).
            # If we have at least 2 such nodes i, j that don't conflict with each other,
            # we can remove k and add i, j.
            
            # Let's implement this.
            # For each k in mask:
            #   candidates = []
            #   for i not in mask:
            #      if (conflicts[i] & mask) == (1 << k):
            #          candidates.append(i)
            #   if len(candidates) >= 2:
            #      # Check if any pair in candidates is non-conflicting
            #      found = False
            #      for a in range(len(candidates)):
            #          for b in range(a+1, len(candidates)):
            #              i = candidates[a]
            #              j = candidates[b]
            #              if (conflicts[i] & (1 << j)) == 0:
            #                  # Remove k, add i, j
            #                  new_mask = (mask & ~(1 << k)) | (1 << i) | (1 << j)
            #                  mask = new_mask
            #                  improved = True
            #                  found = True
            #                  break
            #          if found: break
            #      if found: break # Restart loop or continue?
            
            # This is O(|mask| * nv * nv).
            # |mask| ~ 100. nv ~ 200. 100 * 200 * 200 = 4,000,000 ops.
            # Python is slow. 4M ops per iteration might be too slow if many iterations.
            # But we only do this if simple moves fail.
            
            # To optimize:
            # Only check candidates for a few k's?
            # Or limit the number of candidates checked.
            
            # Let's try a simpler version:
            # Randomly pick a k in mask.
            # Find candidates for k.
            # If >= 2, try to add 2.
            
            if not improved:
                # Try 2-1 swap
                chosen_indices = [i for i in range(nv) if mask & (1 << i)]
                if chosen_indices:
                    random.shuffle(chosen_indices)
                    # Try a few random k's
                    for _ in range(10):
                        if time.time() > deadline: break
                        k = random.choice(chosen_indices)
                        candidates = []
                        for i in range(nv):
                            if mask & (1 << i):
                                continue
                            if (conflicts[i] & mask) == (1 << k):
                                candidates.append(i)
                        if len(candidates) >= 2:
                            # Check pairs
                            # Limit checks to avoid slowdown
                            for a in range(min(5, len(candidates))):
                                for b in range(a+1, min(5, len(candidates))):
                                    i = candidates[a]
                                    j = candidates[b]
                                    if (conflicts[i] & (1 << j)) == 0:
                                        mask = (mask & ~(1 << k)) | (1 << i) | (1 << j)
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                            
        return mask

    best_mask = 0
    best_count = 0
    
    random.seed(42)
    
    # 1. Heuristic Orderings
    order_by_degree = sorted(range(nv), key=lambda i: popcount(conflicts[i]))
    mask = greedy_mis(order_by_degree)
    mask = local_search(mask, max_iters=1000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
        
    order_by_degree_desc = sorted(range(nv), key=lambda i: popcount(conflicts[i]), reverse=True)
    mask = greedy_mis(order_by_degree_desc)
    mask = local_search(mask, max_iters=1000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
        
    # Natural
    mask = greedy_mis(list(range(nv)))
    mask = local_search(mask, max_iters=1000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask

    # 2. Random Restarts
    while time.time() < deadline - 5:
        order = list(range(nv))
        random.shuffle(order)
        mask = greedy_mis(order)
        mask = local_search(mask, max_iters=500)
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
        else:
            # Perturb best
            if random.random() < 0.2:
                # Remove 1-3 random nodes from best
                perturbed = best_mask
                chosen_indices = [i for i in range(nv) if best_mask & (1 << i)]
                if chosen_indices:
                    random.shuffle(chosen_indices)
                    for idx in chosen_indices[:random.randint(1, 3)]:
                        perturbed &= ~(1 << idx)
                    # Re-run local search on perturbed
                    new_mask = local_search(perturbed, max_iters=500)
                    nc = popcount(new_mask)
                    if nc > best_count:
                        best_count = nc
                        best_mask = new_mask

    result = mask_to_set(best_mask)
    if not result:
        result = {valid_nodes[0]}
        
    return list(result)
