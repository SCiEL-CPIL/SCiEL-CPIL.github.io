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
    
    # Build adjacency list for faster degree computation
    adj_list = [[] for _ in range(N + 1)]
    for a in valid_nodes:
        mask = conflict_mask[a]
        for b in valid_nodes:
            if (mask >> b) & 1:
                adj_list[a].append(b)
        
    def greedy_min_degree_init(seed):
        """Greedy independent set using min-degree heuristic."""
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = []
        chosen_mask = 0
        
        while remaining:
            # Find node with minimum degree in remaining
            min_deg = float('inf')
            candidates_at_min = []
            for a in remaining:
                deg = sum(1 for b in remaining if b != a and (conflict_mask[a] >> b) & 1)
                if deg < min_deg:
                    min_deg = deg
                    candidates_at_min = [a]
                elif deg == min_deg:
                    candidates_at_min.append(a)
            
            if candidates_at_min:
                min_node = rng.choice(candidates_at_min)
                chosen.append(min_node)
                chosen_mask |= (1 << min_node)
                remaining.remove(min_node)
                to_remove = [b for b in remaining if (conflict_mask[min_node] >> b) & 1]
                for b in to_remove:
                    remaining.discard(b)
        
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
            # Extract vertices from mask
            verts = []
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                if v >= 1:
                    verts.append(v)
                temp &= temp - 1
            
            if not verts:
                break
                
            for v in verts:
                deg = 0
                for u in verts:
                    if u != v and (conflict_mask[v] >> u) & 1:
                        deg += 1
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v is None:
                break
            
            count += 1
            # Remove best_v and its neighbors from remaining
            remaining &= ~(1 << best_v)
            remaining &= ~conflict_mask[best_v]
        
        return count
    
    def branch_and_bound(candidate_mask, current_size, best_size_ref):
        """
        Branch and bound for maximum independent set.
        candidate_mask: bitmask of vertices still available
        current_size: size of current independent set
        best_size_ref: list [best_size] for mutation
        Returns: (best_mask, best_size)
        """
        if time.time() - start_time > deadline - 5:
            return None, None
        
        if candidate_mask == 0:
            if current_size > best_size_ref[0]:
                best_size_ref[0] = current_size
            return None, None
        
        # Upper bound: current_size + greedy bound on remaining
        ub = current_size + greedy_upper_bound(candidate_mask)
        if ub <= best_size_ref[0]:
            return None, None
        
        # Pick a vertex to branch on (heuristic: pick vertex with max degree in candidate)
        verts = []
        temp = candidate_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            if v >= 1:
                verts.append(v)
            temp &= temp - 1
        
        if not verts:
            return None, None
        
        # Choose pivot vertex
        pivot = None
        max_deg = -1
        for v in verts:
            deg = sum(1 for u in verts if u != v and (conflict_mask[v] >> u) & 1)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            pivot = verts[0]
        
        # Branch 1: Include pivot
        new_mask_include = candidate_mask & ~(1 << pivot) & ~conflict_mask[pivot]
        result_include, size_include = branch_and_bound(new_mask_include, current_size + 1, best_size_ref)
        
        # Branch 2: Exclude pivot
        new_mask_exclude = candidate_mask & ~(1 << pivot)
        result_exclude, size_exclude = branch_and_bound(new_mask_exclude, current_size, best_size_ref)
        
        # Return the better result
        if result_include is not None and (result_exclude is None or size_include >= size_exclude):
            return result_include, size_include
        elif result_exclude is not None:
            return result_exclude, size_exclude
        else:
            return None, None
    
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
        
        # Pick pivot
        verts = []
        temp = candidate_mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            if v >= 1:
                verts.append(v)
            temp &= temp - 1
        
        if not verts:
            return
        
        pivot = None
        max_deg = -1
        for v in verts:
            deg = sum(1 for u in verts if u != v and (conflict_mask[v] >> u) & 1)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            pivot = verts[0]
        
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
    
    # Now use branch and bound to try to improve
    # Start from the greedy solution and try to add more vertices
    remaining_mask = valid_mask_all & ~best_mask
    
    # Use B&B to find a better solution starting from empty, but with the greedy as lower bound
    best_ref = [best_size, best_mask]
    
    # Try B&B from scratch with the greedy lower bound
    if time.time() - start_time < deadline - 10:
        branch_and_bound_with_tracking(valid_mask_all, 0, 0, best_ref)
    
    best_size = best_ref[0]
    best_mask = best_ref[1]
    
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
