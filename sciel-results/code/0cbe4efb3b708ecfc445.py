def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Greedy init
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
            
    best_length = len(values)
    best_sequence = list(values)
    
    current_seq = []
    
    def dfs(products_set, suffix_prods, last_val):
        nonlocal best_length, best_sequence, deadline
        
        if time.time() > deadline:
            return
            
        if len(current_seq) > best_length:
            best_length = len(current_seq)
            best_sequence = list(current_seq)
            
        if len(current_seq) + (128 - last_val) <= best_length:
            return
            
        for x in range(last_val + 1, 129):
            new_suffixes = [x]
            for sp in suffix_prods:
                new_suffixes.append(x * sp)
                
            has_collision = False
            for p in new_suffixes:
                if p in products_set:
                    has_collision = True
                    break
            
            if has_collision:
                continue
                
            products_set.update(new_suffixes)
            current_seq.append(x)
            
            dfs(products_set, new_suffixes, x)
            
            current_seq.pop()
            for p in new_suffixes:
                products_set.discard(p)
                
            if time.time() > deadline:
                return

    dfs(set(), [], 1)
    
    return best_sequence
