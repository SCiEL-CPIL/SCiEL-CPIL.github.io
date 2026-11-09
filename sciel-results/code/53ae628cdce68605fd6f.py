import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Leave 8s margin for safety
    
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
    node_set = set(valid_nodes)
    nv = len(valid_nodes)
    
    # Build conflict graph using bitsets for speed
    # conflicts[i] is a bitmask of valid nodes that conflict with valid_nodes[i]
    conflicts = [0] * nv
    for i_idx in range(nv):
        v = valid_nodes[i_idx]
        for i_j in range(i_idx + 1, nv):
            u = valid_nodes[i_j]
            if v + u in squares:
                conflicts[i_idx] |= (1 << i_j)
                conflicts[i_j] |= (1 << i_idx)
    
    # Greedy MIS using bitsets
    def greedy_mis_bitset(order_indices):
        chosen_mask = 0
        for idx in order_indices:
            # Check if idx conflicts with any chosen
            if (conflicts[idx] & chosen_mask) == 0:
                chosen_mask |= (1 << idx)
        return chosen_mask
    
    def mask_to_set(mask):
        s = set()
        for i in range(nv):
            if mask & (1 << i):
                s.add(valid_nodes[i])
        return s
    
    def popcount(x):
        return bin(x).count('1')
    
    # Enhanced Local Search with 1-swap and 2-swap moves
    def local_search_bitset(mask, max_iters=2000):
        mask = mask
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            # 1. Try adding each non-chosen node (0-swap)
            for i in range(nv):
                if mask & (1 << i):
                    continue
                if (conflicts[i] & mask) == 0:
                    mask |= (1 << i)
                    improved = True
            
            # 2. Try 1-swap: remove one node r, add one node i
            # We only do this if 0-swap didn't improve, or we can combine? 
            # Usually, if we can add, we should. But sometimes swapping is needed.
            # Let's do 1-swap if no 0-swap was found in this iteration pass? 
            # Actually, standard approach: try all moves. If any improves, take it.
            # But 0-swap is strictly better than 1-swap (size +1 vs size 0 change? No, 1-swap is size 0 change? No, 1-swap removes 1 adds 1, size unchanged. 0-swap adds 1, size +1).
            # So we should prioritize 0-swap. If no 0-swap, try 1-swap? No, 1-swap doesn't increase size.
            # Wait, the goal is to MAXIMIZE cardinality.
            # 0-swap: size + 1.
            # 1-swap: size unchanged. Useful to move to a different local optimum that might allow future 0-swaps.
            # 2-swap: size unchanged (remove 2, add 2). Also useful for navigation.
            
            # Let's refine: 
            # First, exhaust all 0-swaps.
            # Then, if stuck, try 1-swaps and 2-swaps to navigate.
            
            # The loop above handles 0-swaps. If 'improved' is True, we loop again.
            # If 'improved' is False, we are at a local optimum for 0-swaps.
            # Now try 1-swaps and 2-swaps to escape.
            
            if not improved:
                # Try 1-swaps
                # For each i not in mask, for each r in mask such that i conflicts with r
                # If we remove r, does i conflict with any other in mask?
                # If not, we can swap. This keeps size same.
                # How to choose which swap? Maybe random or first found.
                # To avoid oscillation, we should be careful.
                # Let's try to find a swap that leads to a state with more "potential" (e.g. lower degree sum)?
                # Or just random restarts.
                
                # Let's try 2-swaps directly as they are more powerful.
                # Pick two random nodes a, b from mask. Remove them.
                # Try to add two new nodes c, d.
                # This is expensive. 
                
                # Alternative: Simulated Annealing or Random Perturbation.
                # Given the time limit, let's stick to a robust 0-swap + random restart strategy.
                # The current code does 0-swap and 1-swap (in the original code, it tried removing conflicts).
                # Actually, the original code's "swap" was: try adding i, if conflict, remove one conflict.
                # This is effectively a 1-swap (remove 1, add 1).
                
                # Let's implement a proper 2-swap:
                # Randomly pick 2 nodes from mask, remove them.
                # Then try to add nodes greedily until no more can be added.
                # If size > current, accept.
                
                # This is a "kick" move.
                
                # Let's implement a function to do a "kick" and return the new mask
                def do_kick(current_mask):
                    # Remove 2 random nodes
                    chosen_indices = [i for i in range(nv) if current_mask & (1 << i)]
                    if len(chosen_indices) < 2:
                        return current_mask
                    random.shuffle(chosen_indices)
                    r1 = chosen_indices[0]
                    r2 = chosen_indices[1]
                    
                    new_mask = current_mask & ~(1 << r1) & ~(1 << r2)
                    
                    # Greedily add nodes
                    # Order by degree? Or random?
                    # Let's try adding in order of ascending degree (least constrained)
                    order = sorted(range(nv), key=lambda x: popcount(conflicts[x]))
                    for i in order:
                        if new_mask & (1 << i):
                            continue
                        if (conflicts[i] & new_mask) == 0:
                            new_mask |= (1 << i)
                            
                    return new_mask

                # Perform a kick
                new_mask = do_kick(mask)
                if popcount(new_mask) > popcount(mask):
                    mask = new_mask
                    improved = True
                    # Continue searching from this new state
                    continue
                else:
                    # If kick didn't improve, we might be stuck.
                    # In this local search function, if we can't improve with 0-swaps and a kick doesn't improve,
                    # we should probably return to the caller to try a different initial state or kick again.
                    # But since we are inside a loop, we can just break and let the outer loop handle it.
                    # Or we can try multiple kicks.
                    # For efficiency, let's break and return. The outer loop will restart.
                    break

        return mask
    
    best_mask = 0
    best_count = 0
    
    # Try multiple random orderings
    random.seed(12345)
    
    # First, try several heuristic orderings
    # Order by degree (ascending)
    order_by_degree = sorted(range(nv), key=lambda i: popcount(conflicts[i]))
    mask = greedy_mis_bitset(order_by_degree)
    mask = local_search_bitset(mask, max_iters=1000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
    
    # Order by degree (descending)
    order_by_degree_desc = sorted(range(nv), key=lambda i: popcount(conflicts[i]), reverse=True)
    mask = greedy_mis_bitset(order_by_degree_desc)
    mask = local_search_bitset(mask, max_iters=1000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
    
    # Natural order
    mask = greedy_mis_bitset(list(range(nv)))
    mask = local_search_bitset(mask, max_iters=1000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
    
    # Reverse natural order
    mask = greedy_mis_bitset(list(range(nv - 1, -1, -1)))
    mask = local_search_bitset(mask, max_iters=1000)
    c = popcount(mask)
    if c > best_count:
        best_count = c
        best_mask = mask
    
    # Random orderings with local search
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        # Strategy: Start from best_mask, perturb, and local search
        # Or start from scratch?
        # Starting from scratch with random order is good for diversity.
        # Starting from best is good for refinement.
        
        if attempt % 2 == 0:
            # Random start
            order = list(range(nv))
            random.shuffle(order)
            mask = greedy_mis_bitset(order)
        else:
            # Perturb best
            mask = best_mask
            # Remove 2-3 random nodes
            chosen_indices = [i for i in range(nv) if mask & (1 << i)]
            if chosen_indices:
                random.shuffle(chosen_indices)
                for idx in chosen_indices[:2]:
                    mask &= ~(1 << idx)
                # Greedily fill
                order = list(range(nv))
                random.shuffle(order)
                for idx in order:
                    if (conflicts[idx] & mask) == 0:
                        mask |= (1 << idx)
        
        mask = local_search_bitset(mask, max_iters=500)
        c = popcount(mask)
        if c > best_count:
            best_count = c
            best_mask = mask
    
    # Final polish: Intensive local search on best
    if time.time() < deadline - 2:
        mask = best_mask
        for _ in range(100):
            if time.time() > deadline:
                break
            # Try to improve with a longer local search
            mask = local_search_bitset(mask, max_iters=1000)
            c = popcount(mask)
            if c > best_count:
                best_count = c
                best_mask = mask
            else:
                # If no improvement, try a kick
                chosen_indices = [i for i in range(nv) if best_mask & (1 << i)]
                if len(chosen_indices) >= 2:
                    random.shuffle(chosen_indices)
                    r1 = chosen_indices[0]
                    r2 = chosen_indices[1]
                    perturbed = best_mask & ~(1 << r1) & ~(1 << r2)
                    # Greedily fill
                    order = sorted(range(nv), key=lambda x: popcount(conflicts[x]))
                    for idx in order:
                        if (conflicts[idx] & perturbed) == 0:
                            perturbed |= (1 << idx)
                    
                    perturbed = local_search_bitset(perturbed, max_iters=500)
                    c = popcount(perturbed)
                    if c > best_count:
                        best_count = c
                        best_mask = perturbed
    
    # Convert to list of nodes
    result = mask_to_set(best_mask)
    if not result:
        result = {valid_nodes[0]}
    
    return list(result)
