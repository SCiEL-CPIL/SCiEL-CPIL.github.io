import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    def greedy_from(start_val, current_seq, current_products):
        # current_seq is list
        # current_products is set
        seq = list(current_seq)
        prods = set(current_products)
        
        for x in range(start_val, 129):
            if time.time() > deadline:
                break
            # Check if x can be added
            tails = []
            prod = x
            for a in reversed(seq):
                prod *= a
                tails.append(prod)
            
            # Check collision
            if not prods.intersection(tails):
                seq.append(x)
                prods.update(tails)
                
        return seq, prods

    best_seq = []
    
    # Explore first 12 numbers with backtracking
    # Numbers 2 to 13
    def backtrack(idx, current_seq, current_products):
        nonlocal best_seq
        if time.time() > deadline:
            return
        
        if idx > 13:
            # Greedy for the rest
            seq, _ = greedy_from(idx, current_seq, current_products)
            if len(seq) > len(best_seq):
                best_seq = seq
            return
            
        # Option 1: Skip current number (idx)
        backtrack(idx + 1, current_seq, current_products)
        
        # Option 2: Take current number (idx)
        # Check if idx can be added to current_seq
        # current_seq ends with last < idx
        tails = []
        prod = idx
        for a in reversed(current_seq):
            prod *= a
            tails.append(prod)
            
        if not current_products.intersection(tails):
            new_seq = current_seq + [idx]
            new_prods = current_products.union(tails)
            backtrack(idx + 1, new_seq, new_prods)

    backtrack(2, [], set())
    
    # If best_seq is empty, run simple greedy
    if not best_seq:
        seq, _ = greedy_from(2, [], set())
        best_seq = seq
        
    return best_seq
