import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def calc_suffix(seq):
        s = []
        prod = 1
        for a in reversed(seq):
            prod *= a
            s.append(prod)
        return s

    def greedy():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values, products

    best_sequence, best_products = greedy()
    best_length = len(best_sequence)
    
    def dfs(start_seq, start_products, start_suffix):
        nonlocal best_sequence, best_length, deadline
        
        current_seq = start_seq[:]
        current_products = start_products
        current_suffix = start_suffix[:]
        
        def recurse(start_val):
            nonlocal best_sequence, best_length, deadline, current_seq, current_products, current_suffix
            
            if time.time() > deadline:
                return False
            
            if len(current_seq) > best_length:
                best_length = len(current_seq)
                best_sequence = current_seq[:]
                
            if len(current_seq) + (129 - start_val) <= best_length:
                return True
                
            for x in range(start_val, 129):
                new_suffix = [x]
                prod = x
                for p in current_suffix:
                    prod *= p
                    new_suffix.append(prod)
                    
                valid = True
                for p in new_suffix:
                    if p in current_products:
                        valid = False
                        break
                        
                if not valid:
                    continue
                    
                # Save state
                old_seq_len = len(current_seq)
                current_seq.append(x)
                current_products.update(new_suffix)
                old_suffix = current_suffix
                current_suffix = new_suffix
                
                if not recurse(x + 1):
                    # Timeout
                    current_seq = current_seq[:old_seq_len]
                    current_products.difference_update(new_suffix)
                    current_suffix = old_suffix
                    return False
                    
                # Backtrack
                current_seq = current_seq[:old_seq_len]
                current_products.difference_update(new_suffix)
                current_suffix = old_suffix
                
                if time.time() > deadline:
                    return False
                    
            return True
            
        recurse(start_seq[-1] + 1 if start_seq else 2)

    # Run from Greedy
    suffix = calc_suffix(best_sequence)
    dfs(best_sequence, best_products, suffix)
    
    # Run from short starts if time
    if time.time() < deadline - 15:
        dfs([2], {2}, [2])
        
    if time.time() < deadline - 5:
        dfs([2, 3], {2, 3, 6}, [3, 6])
        
    return best_sequence
