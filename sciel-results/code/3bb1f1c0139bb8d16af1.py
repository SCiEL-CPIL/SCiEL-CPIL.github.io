import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Perfect squares in [2, 512]
    squares = set()
    i = 1
    while i * i <= max_sum:
        if i * i >= 2:
            squares.add(i * i)
        i += 1
    
    # Diagonal constraint: 2a not in squares
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)
    num_cand = len(candidates)
    
    # Build conflict graph using bitmasks for speed
    # Map candidates to indices 0..num_cand-1
    idx_map = {a: i for i, a in enumerate(candidates)}
    
    # Conflict bitmask: for each candidate index, bitmask of conflicting candidate indices
    conflict_mask = [0] * num_cand
    for i, a in enumerate(candidates):
        for j in range(i + 1, num_cand):
            b = candidates[j]
            if a + b in squares:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute popcount for faster degree calculation
    # Using a lookup table for bytes
    popcount_table = [0] * 256
    for i in range(256):
        popcount_table[i] = bin(i).count('1')
    
    def popcount(x):
        """Fast popcount using table lookup"""
        count = 0
        while x:
            count += popcount_table[x & 0xFF]
            x >>= 8
        return count
    
    # Greedy coloring for upper bound on independent set
    def greedy_color_bound(node_mask):
        """
        node_mask: bitmask of nodes to consider
        Returns an upper bound on the size of the maximum independent set in node_mask.
        Uses greedy coloring on the conflict graph restricted to node_mask.
        """
        if node_mask == 0:
            return 0
        
        uncolored = node_mask
        color_count = 0
        
        while uncolored:
            # Find a maximal independent set in uncolored (greedy)
            # This becomes one color class
            color_class = 0
            remaining = uncolored
            while remaining:
                # Pick a node with smallest degree in remaining
                # For speed, just pick the lowest set bit
                v = remaining & (-remaining)  # lowest set bit
                vi = v.bit_length() - 1
                color_class |= v
                # Remove v and its neighbors from remaining
                remaining &= ~(v | (conflict_mask[vi] & remaining))
            
            uncolored &= ~color_class
            color_count += 1
        
        return color_count
    
    # Branch and bound for maximum independent set
    # Uses bitmasks for speed
    best_size = [0]
    best_set = [0]  # bitmask of best solution
    
    def branch_and_bound(node_mask, current_mask, current_size):
        if time.time() - start_time > deadline - 2.0:
            return
        
        if node_mask == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_set[0] = current_mask
            return
        
        # Upper bound: current_size + greedy coloring of node_mask
        ub = current_size + greedy_color_bound(node_mask)
        if ub <= best_size[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in node_mask
        # Sample a few vertices to find one with high degree
        best_v = None
        best_deg = -1
        temp = node_mask
        # Check up to 8 vertices for good branching candidate
        for _ in range(8):
            if not temp:
                break
            v = temp & (-temp)
            vi = v.bit_length() - 1
            deg = popcount(conflict_mask[vi] & node_mask)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp &= ~v
        
        if best_v is None:
            best_v = node_mask & (-node_mask)
        
        vi = best_v.bit_length() - 1
        
        # Branch 1: include vi
        new_node_mask = node_mask & ~(best_v | conflict_mask[vi])
        new_current_mask = current_mask | best_v
        branch_and_bound(new_node_mask, new_current_mask, current_size + 1)
        
        # Branch 2: exclude vi
        new_node_mask2 = node_mask & ~best_v
        branch_and_bound(new_node_mask2, current_mask, current_size)
    
    # Greedy initial solution using bitmask
    def greedy_solution():
        remaining = (1 << num_cand) - 1  # all candidates
        selected = 0
        while remaining:
            # Pick node with smallest degree in remaining
            best_v = None
            best_deg = float('inf')
            temp = remaining
            while temp:
                v = temp & (-temp)
                vi = v.bit_length() - 1
                deg = popcount(conflict_mask[vi] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                temp &= ~(v)
            
            if best_v is None:
                break
            
            selected |= best_v
            vi = best_v.bit_length() - 1
            remaining &= ~(best_v | conflict_mask[vi])
        
        return selected
    
    # Convert bitmask to set of candidate values
    def mask_to_set(mask):
        result = set()
        i = 0
        while mask:
            if mask & 1:
                result.add(candidates[i])
            mask >>= 1
            i += 1
        return result
    
    # Local search with swap moves
    def local_search(initial_mask):
        current_mask = initial_mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size_local = current_size
        
        for iteration in range(100):
            if time.time() - start_time > deadline - 3.0:
                break
            
            improved = False
            
            # Try to add a vertex
            for i in range(num_cand):
                v = 1 << i
                if not (current_mask & v):
                    # Check if v can be added
                    if not (conflict_mask[i] & current_mask):
                        current_mask |= v
                        current_size += 1
                        improved = True
                        if current_size > best_size_local:
                            best_size_local = current_size
                            best_mask = current_mask
                        break
            
            if improved:
                continue
            
            # Try swap: remove one, add one or more
            for i in range(num_cand):
                v = 1 << i
                if current_mask & v:
                    temp_mask = current_mask & ~v
                    temp_size = current_size - 1
                    # Try to add as many as possible
                    for j in range(num_cand):
                        w = 1 << j
                        if not (temp_mask & w) and not (conflict_mask[j] & temp_mask):
                            temp_mask |= w
                            temp_size += 1
                    
                    if temp_size > current_size:
                        current_mask = temp_mask
                        current_size = temp_size
                        improved = True
                        if current_size > best_size_local:
                            best_size_local = current_size
                            best_mask = current_mask
                        break
            
            if not improved:
                break
        
        return best_mask, best_size_local
    
    # Start with greedy
    greedy_mask = greedy_solution()
    greedy_size = popcount(greedy_mask)
    best_mask = greedy_mask
    best_size[0] = greedy_size
    
    # Randomized restarts
    for seed in range(300):
        if time.time() - start_time > deadline - 15.0:
            break
        random.seed(seed)
        remaining = (1 << num_cand) - 1
        selected = 0
        while remaining:
            # Pick node with smallest degree, with random tie-breaking
            min_deg = float('inf')
            min_nodes = []
            temp = remaining
            while temp:
                v = temp & (-temp)
                vi = v.bit_length() - 1
                deg = popcount(conflict_mask[vi] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
                temp &= ~(v)
            
            if not min_nodes:
                break
            
            v = random.choice(min_nodes)
            selected |= v
            vi = v.bit_length() - 1
            remaining &= ~(v | conflict_mask[vi])
        
        sel_size = popcount(selected)
        if sel_size > best_size[0]:
            best_size[0] = sel_size
            best_mask = selected
    
    # Local search on best
    best_mask, best_size[0] = local_search(best_mask)
    
    # Try B&B with time limit
    remaining_time = deadline - 5.0 - (time.time() - start_time)
    if remaining_time > 3.0:
        # Reset best for B&B
        best_size[0] = popcount(best_mask)
        best_set[0] = best_mask
        
        # Run B&B
        all_nodes = (1 << num_cand) - 1
        branch_and_bound(all_nodes, 0, 0)
        
        if best_set[0]:
            bbb_size = popcount(best_set[0])
            if bbb_size > popcount(best_mask):
                best_mask = best_set[0]
                best_size[0] = bbb_size
    
    # Final local search
    best_mask, final_size = local_search(best_mask)
    
    # Convert to set of values
    result_set = mask_to_set(best_mask)
    
    # Final verification
    for a in result_set:
        if 2 * a in squares:
            return []
    for i, a in enumerate(result_set):
        for b in result_set:
            if a < b and a + b in squares:
                return []
    
    return list(result_set)
