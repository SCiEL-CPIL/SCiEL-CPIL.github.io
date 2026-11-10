import time
def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 60s. We stop at 50s.
    deadline = start_time + 50
    n = 256
    
    # 1. Precompute conflict masks
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                
    # 2. Initial Heuristic to get a good lower bound (incumbent)
    # Greedy by conflict count (lowest first)
    order = sorted(range(1, n + 1), key=lambda x: conflict_mask[x].bit_count())
    init_mask = 0
    for v in order:
        if (conflict_mask[v] & init_mask) == 0:
            init_mask |= (1 << v)
            
    best_size = init_mask.bit_count()
    best_mask = init_mask

    # 3. Branch and Bound
    # We use a non-recursive or carefully managed recursive approach.
    # Python recursion limit might be an issue, but depth is max 256.
    # However, Python function calls are slow. We will use a stack-based or optimized recursive call.
    # Given N=256, recursion depth is fine.
    
    def get_color_bound(cand_mask):
        """
        Returns an upper bound on the independence number of the subgraph defined by cand_mask.
        Uses greedy coloring.
        """
        if not cand_mask:
            return 0
        
        remaining = cand_mask
        color_count = 0
        
        while remaining:
            color_count += 1
            
            # Find vertex with max degree in remaining
            best_v = -1
            max_deg = -1
            temp = remaining
            while temp:
                bit = temp & (-temp)
                v = bit.bit_length() - 1
                # Degree in remaining
                deg = (conflict_mask[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= bit
            
            if best_v == -1:
                break
                
            # Remove best_v and its neighbors
            remaining &= ~(1 << best_v)
            remaining &= ~conflict_mask[best_v]
            
            # Optimization: If color_count + current_size_in_caller <= best_size, we could stop?
            # We don't have current_size here. We just return the count.
            
        return color_count

    def bb(candidates, current_mask, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
            
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Upper bound
        # Note: get_color_bound is expensive. We call it every node.
        # If it's too slow, we can cache or use a simpler bound.
        # For now, let's try it.
        ub = current_size + get_color_bound(candidates)
        
        if ub <= best_size:
            return
            
        # Select pivot: Max degree in candidates
        pivot = -1
        max_deg = -1
        temp = candidates
        while temp:
            bit = temp & (-temp)
            v = bit.bit_length() - 1
            deg = (conflict_mask[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp ^= bit
            
        if pivot == -1:
            return
            
        # Branch 1: Include pivot
        # Remove pivot and neighbors from candidates
        new_cand = candidates & ~(1 << pivot) & ~conflict_mask[pivot]
        bb(new_cand, current_mask | (1 << pivot), current_size + 1)
        
        # Check time again before second branch
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude pivot
        new_cand = candidates & ~(1 << pivot)
        bb(new_cand, current_mask, current_size)

    # Run B&B
    all_candidates = (1 << (n + 1)) - 2 # Bits 1 to n set
    # We start with empty set, but we have a good incumbent.
    # To make B&B faster, we can start from the incumbent?
    # No, standard B&B starts from empty. The incumbent just helps pruning.
    
    # Optimization: If we have a strong incumbent, we can prune early.
    bb(all_candidates, 0, 0)
    
    # If B&B didn't finish or didn't improve, we might have a good heuristic result.
    # But B&B explores the whole space (pruned).
    
    # Convert best_mask to list
    result = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    
    return result
