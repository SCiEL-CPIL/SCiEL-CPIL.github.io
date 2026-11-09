import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    # Build conflict graph using bitmasks for speed
    # adj_mask[i] is a bitmask of nodes conflicting with i
    adj_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
    
    best_size = 0
    best_set = 0
    
    # Greedy initialization
    # Heuristic: pick node with lowest degree in remaining candidates
    candidates = (1 << (n + 1)) - 2
    selected = 0
    temp = candidates
    while temp:
        min_deg = float('inf')
        chosen = -1
        t = temp
        while t:
            lsb = t & (-t)
            node = lsb.bit_length() - 1
            deg = (adj_mask[node] & temp).bit_count()
            if deg < min_deg:
                min_deg = deg
                chosen = node
            t &= t - 1
        
        if chosen == -1:
            break
        
        selected |= (1 << chosen)
        temp &= ~(adj_mask[chosen] | (1 << chosen))
    
    best_size = selected.bit_count()
    best_set = selected
    
    def branch_and_bound(candidates_mask, current_size, selected_mask):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = selected_mask
            return
        
        # Upper bound: current_size + number of candidates
        # This is a valid upper bound
        ub = current_size + candidates_mask.bit_count()
        if ub <= best_size:
            return
        
        # Pick a node to branch on
        # Heuristic: Highest degree in candidate subgraph
        max_deg = -1
        chosen_node = -1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            node = lsb.bit_length() - 1
            deg = (adj_mask[node] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                chosen_node = node
            temp &= temp - 1
        
        if chosen_node == -1:
            return
        
        chosen_bit = 1 << chosen_node
        
        # Branch 1: Include chosen_node
        new_candidates = candidates_mask & ~(adj_mask[chosen_node] | chosen_bit)
        new_selected = selected_mask | chosen_bit
        branch_and_bound(new_candidates, current_size + 1, new_selected)
        
        # Branch 2: Exclude chosen_node
        new_candidates = candidates_mask & ~chosen_bit
        branch_and_bound(new_candidates, current_size, selected_mask)

    all_nodes_mask = (1 << (n + 1)) - 2
    branch_and_bound(all_nodes_mask, 0, 0)
    
    # Convert best_set bitmask to list
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << i):
            result.append(i)
    
    return result if result else [1]
