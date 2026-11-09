import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Precompute forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency as bitmasks
    # adj[u] is a bitmask where bit v is 1 if u and v conflict
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask
        
    # Helper: Check if a vertex can be added to a set (bitmask)
    def can_add(v, set_mask):
        return (set_mask & adj[v]) == 0

    def greedy_fill(base_mask):
        """
        Greedily add vertices to base_mask in a randomized order.
        Returns the maximal independent set mask.
        """
        current = base_mask
        # Get list of candidates not in current
        candidates = [i for i in range(m) if not (current >> i) & 1]
        random.shuffle(candidates)
        
        for v in candidates:
            if can_add(v, current):
                current |= (1 << v)
                
        return current

    def local_search_improve(current_mask):
        """
        Improve the independent set using local moves.
        Moves:
        1. Add any compatible vertex.
        2. Swap 1 vertex with 1 vertex if it leads to a larger set after greedy fill.
        3. Swap 2 vertices with 2 or 3 vertices if it leads to a larger set.
        """
        improved = True
        while improved:
            improved = False
            size = current_mask.bit_count()
            
            # 1. Try to add any vertex
            # Find any v not in set such that can_add(v, set)
            # We can just call greedy_fill, but that might add multiple.
            # If greedy_fill adds anything, we accept and loop again.
            new_mask = greedy_fill(current_mask)
            if new_mask != current_mask:
                current_mask = new_mask
                improved = True
                continue
            
            # 2. Try 1-for-1 swaps
            verts_in_set = [i for i in range(m) if (current_mask >> i) & 1]
            verts_not_in_set = [i for i in range(m) if not (current_mask >> i) & 1]
            
            # To be efficient, we only try a subset of swaps or all if small
            # Here N=205, so sets are small (size ~5-10).
            # Number of pairs is ~10 * 195 = 1950. This is fast.
            
            found_swap = False
            # We want to find a swap that leads to a larger set.
            # We try all valid swaps, run greedy fill, and keep the best result.
            best_swap_mask = current_mask
            best_swap_size = size
            
            for v_out in verts_in_set:
                temp_mask = current_mask & ~(1 << v_out)
                # Find all v_in that can be added to temp_mask
                for v_in in verts_not_in_set:
                    if can_add(v_in, temp_mask):
                        candidate_mask = temp_mask | (1 << v_in)
                        # Try to extend
                        extended = greedy_fill(candidate_mask)
                        ext_size = extended.bit_count()
                        if ext_size > best_swap_size:
                            best_swap_size = ext_size
                            best_swap_mask = extended
                            found_swap = True
                            
            if found_swap:
                current_mask = best_swap_mask
                improved = True
                continue
                
            # 3. Try 2-for-2 or 2-for-3 swaps
            # This is more expensive. O(|S|^2 * |V\S|^2)
            # If |S| is ~10, |S|^2 is 100. |V\S| is ~195. 195^2 is ~40k.
            # 100 * 40k = 4M operations. Might be too slow per step.
            # We can sample or limit this.
            
            # Let's try a limited number of random 2-swaps
            if len(verts_in_set) >= 2:
                # Pick 2 random vertices to remove
                # And try to add 2 or 3
                # We will do this a few times
                for _ in range(50):
                    if not verts_in_set or len(verts_in_set) < 2:
                        break
                    v1, v2 = random.sample(verts_in_set, 2)
                    temp_mask = current_mask & ~(1 << v1) & ~(1 << v2)
                    
                    # Find candidates to add
                    # We want to add at least 2 to improve size, or 3.
                    # Let's just try to greedily fill from temp_mask and see if it's better.
                    # But greedy fill might not add the specific 2 we removed.
                    # Better: try to add specific pairs/triples?
                    # Simpler: Just run greedy_fill on temp_mask. 
                    # If the result is larger, accept.
                    # This is equivalent to "remove 2, refill greedily".
                    extended = greedy_fill(temp_mask)
                    if extended.bit_count() > size:
                        current_mask = extended
                        improved = True
                        break
                        
        return current_mask

    best_size = 0
    best_set_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial Greedy Runs
    for _ in range(500):
        if time.time() > deadline:
            break
        sol = greedy_fill(0)
        size = sol.bit_count()
        if size > best_size:
            best_size = size
            best_set_mask = sol
            
        improved_sol = local_search_improve(sol)
        if improved_sol.bit_count() > best_size:
            best_size = improved_sol.bit_count()
            best_set_mask = improved_sol

    # Main Loop: Kick-based Local Search
    while time.time() < deadline:
        # Perturb best solution
        # Remove 1 to 3 random vertices
        num_removals = random.randint(1, 3)
        current_mask = best_set_mask
        
        verts_in_set = [i for i in range(m) if (current_mask >> i) & 1]
        
        if len(verts_in_set) < num_removals:
            new_mask = greedy_fill(0)
        else:
            to_remove = random.sample(verts_in_set, num_removals)
            for v in to_remove:
                current_mask &= ~(1 << v)
            
            new_mask = greedy_fill(current_mask)
            new_mask = local_search_improve(new_mask)
            
        if new_mask.bit_count() > best_size:
            best_size = new_mask.bit_count()
            best_set_mask = new_mask
            
    # Extract result
    result = []
    for i in range(m):
        if (best_set_mask >> i) & 1:
            result.append(i)
            
    return np.array(result, dtype=np.int64)
