import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def is_valid_append(seq, products, suffixes, x):
        # New suffixes: [x, x*suffixes[0], x*suffixes[1], ...]
        new_suffixes = [x]
        prod = x
        
        # Check collision with existing products
        # We can check as we generate to fail fast
        if x in products:
            return None, None
            
        for s in suffixes:
            prod *= s
            new_suffixes.append(prod)
            if prod in products:
                return None, None
        
        # If we reach here, no collision with old products
        # New products are distinct from each other (proven)
        new_products = products.union(new_suffixes)
        return new_products, new_suffixes

    def fast_greedy():
        values = []
        products = set()
        suffixes = []
        for x in range(2, 129):
            new_prods, new_sufs = is_valid_append(values, products, suffixes, x)
            if new_prods is not None:
                values.append(x)
                products = new_prods
                suffixes = new_sufs
        return values

    best_seq = fast_greedy()
    best_len = len(best_seq)
    
    # Beam Search
    K = 500  # Beam width
    W = 30   # Candidate window
    
    beam = [([], set(), [])]
    
    while time.time() < deadline:
        new_beam = []
        
        for seq, products, suffixes in beam:
            if time.time() > deadline:
                break
                
            if not seq:
                last = 1
            else:
                last = seq[-1]
                
            # Prune: If even taking all remaining numbers can't beat best_len, skip?
            if len(seq) + (128 - last) < best_len:
                continue
                
            # Generate candidates
            start_x = last + 1
            end_x = min(last + W, 128)
            
            for x in range(start_x, end_x + 1):
                new_prods, new_sufs = is_valid_append(seq, products, suffixes, x)
                if new_prods is not None:
                    new_seq = seq + [x]
                    new_beam.append((new_seq, new_prods, new_sufs))
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                        
        if not new_beam:
            break
            
        # Sort and Prune
        def sort_key(state):
            seq = state[0]
            last = seq[-1] if seq else 0
            s = sum(seq)
            return (-len(seq), last, s)
            
        new_beam.sort(key=sort_key)
        
        # Keep top K
        beam = new_beam[:K]
        
        # Early stopping
        if beam:
            top_seq = beam[0][0]
            if top_seq:
                max_possible = len(top_seq) + (128 - top_seq[-1])
            else:
                max_possible = 128
                
            if max_possible < best_len:
                break

    return best_seq
