import time

def construct():
    start_time = time.time()
    # Use 55s to be safe within the 60s limit
    deadline = start_time + 55.0

    best_seq = []
    best_len = 0

    # Helper to check if adding x to current_seq creates unique products
    # and returns the list of new products if valid, else None
    def try_add(current_seq, x, current_products):
        # Calculate new products: x, x*last, x*last*second_last, etc.
        # We check for collisions on the fly to avoid list creation if possible
        # But we need the list to add/remove from the set later.
        # So we compute the list first.
        
        new_prods = []
        prod = x
        new_prods.append(prod)
        
        # Calculate products of x * suffix of current_seq
        for a in reversed(current_seq):
            prod *= a
            new_prods.append(prod)
            
        # Check for collisions with existing products
        for p in new_prods:
            if p in current_products:
                return None # Collision
                
        return new_prods

    def dfs(last_picked, current_seq, current_products):
        nonlocal best_seq, best_len

        if time.time() > deadline:
            return

        current_len = len(current_seq)
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_seq = current_seq[:]

        # Pruning: Max possible extension
        # Remaining numbers are last_picked+1 to 128.
        # Count is 128 - last_picked.
        if current_len + (128 - last_picked) <= best_len:
            return

        if last_picked >= 128:
            return

        # Try adding next numbers
        for x in range(last_picked + 1, 129):
            if time.time() > deadline:
                return

            new_prods = try_add(current_seq, x, current_products)
            
            if new_prods is not None:
                # Add x
                current_seq.append(x)
                for p in new_prods:
                    current_products.add(p)
                
                dfs(x, current_seq, current_products)
                
                # Undo
                current_seq.pop()
                for p in new_prods:
                    current_products.remove(p)
                    
                if time.time() > deadline:
                    return

    # 1. Greedy initialization to get a good lower bound quickly
    greedy_seq = []
    greedy_prods = set()
    for x in range(2, 129):
        if time.time() > deadline:
            break
        new_prods = try_add(greedy_seq, x, greedy_prods)
        if new_prods is not None:
            greedy_seq.append(x)
            for p in new_prods:
                greedy_prods.add(p)
    
    best_seq = greedy_seq
    best_len = len(greedy_seq)
    
    # 2. DFS to find optimal solution
    dfs(1, [], set())
    
    return best_seq
