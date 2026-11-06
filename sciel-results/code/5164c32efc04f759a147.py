import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict_mask[i] has bit j set if i and j conflict (i+j is square)
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        mask = 0
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                mask |= (1 << b)
        conflict_mask[a] = mask
        
    # Pre-filter: nodes where 2a is a square are invalid
    valid_nodes = []
    valid_mask_all = 0
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            valid_mask_all |= (1 << a)
            
    for a in range(1, N + 1):
        conflict_mask[a] &= valid_mask_all
    
    def popcount(x):
        """Count bits in integer x."""
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def greedy_min_degree_init(seed):
        """Greedy independent set using min-degree heuristic with bit operations."""
        rng = random.Random(seed)
        remaining = valid_mask_all
        chosen = []
        chosen_mask = 0
        
        while remaining:
            # Find node with minimum degree in remaining using bit ops
            min_deg = float('inf')
            candidates_at_min = []
            temp = remaining
            while temp:
                a = (temp & -temp).bit_length() - 1
                if a >= 1:
                    deg = popcount(conflict_mask[a] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        candidates_at_min = [a]
                    elif deg == min_deg:
                        candidates_at_min.append(a)
                temp &= temp - 1
            
            if candidates_at_min:
                min_node = rng.choice(candidates_at_min)
                chosen.append(min_node)
                chosen_mask |= (1 << min_node)
                # Remove min_node and its neighbors
                remaining &= ~(1 << min_node)
                remaining &= ~conflict_mask[min_node]
        
        return chosen_mask, chosen
    
    def greedy_upper_bound(candidate_mask):
        """Compute a greedy independent set size from candidate_mask for upper bound."""
        if candidate_mask == 0:
            return 0
        count = 0
        remaining = candidate_mask
        while remaining:
            # Pick a vertex with minimum degree in remaining
            best_v = None
            best_deg = float('inf')
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                if v >= 1:
                    deg = popcount(conflict_mask[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                temp &= temp - 1
            
            if best_v is None:
                break
            
            count += 1
            # Remove best_v and its neighbors from remaining
            remaining &= ~(1 << best_v)
            remaining &= ~conflict_mask[best_v]
        
        return count
    
    def local_search_improve(mask, max_iters=5000):
        """Try to improve an independent set by adding vertices or swapping."""
        if time.time() - start_time > deadline - 5:
            return mask
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() - start_time < deadline - 5:
            improved = False
            iterations += 1
            # Try to add any vertex not in mask that doesn't conflict
            candidates = valid_mask_all & ~mask
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                if v >= 1:
                    if (conflict_mask[v] & mask) == 0:
                        mask |= (1 << v)
                        improved = True
                    temp &= temp - 1
            # Try swaps: remove one, add two
            if not improved:
                temp = mask
                while temp:
                    a = (temp & -temp).bit_length() - 1
                    if a >= 1:
                        mask_without_a = mask & ~(1 << a)
                        # Find vertices that conflict with a but not with rest
                        new_candidates = valid_mask_all & ~mask_without_a
                        new_candidates &= ~conflict_mask[a]  # can't add neighbors of a
                        temp2 = new_candidates
                        while temp2:
                            b = (temp2 & -temp2).bit_length() - 1
                            if b >= 1 and b != a:
                                if (conflict_mask[b] & mask_without_a) == 0:
                                    # Try adding b
                                    mask_with_b = mask_without_a | (1 << b)
                                    # Now try to add another vertex c
                                    temp3 = valid_mask_all & ~mask_with_b
                                    while temp3:
                                        c = (temp3 & -temp3).bit_length() - 1
                                        if c >= 1 and c != a and c != b:
                                            if (conflict_mask[c] & mask_with_b) == 0:
                                                new_mask = mask_with_b | (1 << c)
                                                if popcount(new_mask) > popcount(mask):
                                                    mask = new_mask
                                                    improved = True
                                                    break
                                        temp3 &= temp3 - 1
                                    if improved:
                                        break
                            temp2 &= temp2 - 1
                        if improved:
                            break
                    temp &= temp - 1
        return mask
    
    def branch_and_bound_with_tracking(candidate_mask, current_set_mask, current_size, best_ref):
        """
        Branch and bound that tracks the actual set mask for reconstruction.
        best_ref: list [best_size, best_mask]
        """
        if time.time() - start_time > deadline - 5:
            return
        
        if candidate_mask == 0:
            if current_size > best_ref[0]:
                best_ref[0] = current_size
                best_ref[1] = current_set_mask
            return
        
        # Upper bound
        ub = current_size + greedy_upper_bound(candidate_mask)
        if ub <= best_ref[0]:
            return
        
        # Pick pivot with max degree
        pivot = None
        max_deg = -1
        temp = candidate_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            if v >= 1:
                deg = popcount(conflict_mask[v] & candidate_mask)
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            temp &= temp - 1
        
        if pivot is None:
            temp = candidate_mask
            if temp:
                pivot = (temp & -temp).bit_length() - 1
        
        if pivot is None:
            return
        
        # Branch 1: Include pivot
        new_mask_include = candidate_mask & ~(1 << pivot) & ~conflict_mask[pivot]
        branch_and_bound_with_tracking(new_mask_include, current_set_mask | (1 << pivot), current_size + 1, best_ref)
        
        # Branch 2: Exclude pivot
        new_mask_exclude = candidate_mask & ~(1 << pivot)
        branch_and_bound_with_tracking(new_mask_exclude, current_set_mask, current_size, best_ref)
    
    # First, get a good initial solution using greedy
    init_mask, init_list = greedy_min_degree_init(0)
    best_size = len(init_list)
    best_mask = init_mask
    best_list = list(init_list)
    
    # Improve with local search
    best_mask = local_search_improve(best_mask)
    best_size = popcount(best_mask)
    
    # Try multiple greedy seeds and keep best
    for seed in range(1, 10):
        if time.time() - start_time > deadline - 15:
            break
        gm, gl = greedy_min_degree_init(seed)
        gm = local_search_improve(gm)
        gs = popcount(gm)
        if gs > best_size:
            best_size = gs
            best_mask = gm
    
    # Use B&B to find a better solution starting from empty, but with the greedy as lower bound
    best_ref = [best_size, best_mask]
    
    # Try B&B from scratch with the greedy lower bound
    if time.time() - start_time < deadline - 10:
        branch_and_bound_with_tracking(valid_mask_all, 0, 0, best_ref)
    
    best_size = best_ref[0]
    best_mask = best_ref[1]
    
    # Final local search polish
    if time.time() - start_time < deadline - 3:
        best_mask = local_search_improve(best_mask)
        best_size = popcount(best_mask)
    
    # Extract the list from best_mask
    best_list = []
    temp = best_mask
    while temp:
        v = (temp & -temp).bit_length() - 1
        if v >= 1:
            best_list.append(v)
        temp &= temp - 1
    
    # Final verification and cleanup
    result = set(best_list)
    for a in result:
        for b in result:
            if a <= b:
                s = a + b
                r = int(math.isqrt(s))
                if r * r == s:
                    # Remove conflicting element (shouldn't happen if B&B is correct)
                    result.discard(b)
                    break
    
    return list(result)
