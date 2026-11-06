import time

def construct():
        start_time = time.time()
        deadline = start_time + 55.0
        best_seq = []
        
        def check_validity(seq, x):
            # Returns (is_valid, new_products_set)
            # seq is the current sequence
            # x is the new number to append
            # We need to check if any block product ending at x collides with existing products
            # Or if x collides with existing products
            
            # To avoid recalculating all products, we can maintain the set.
            # But in DFS, we pass the set.
            pass
        
        def dfs(seq, products_set, last_val):
            nonlocal best_seq
            if time.time() > deadline:
                return
            
            if len(seq) > len(best_seq):
                best_seq = list(seq)
                
            # Try to append next number
            for x in range(last_val + 1, 129):
                # Calculate new products ending at x
                # prod = x
                # for i in reversed(range(len(seq))):
                #    prod *= seq[i]
                #    if prod in products_set: break
                #    ...
                
                # Optimization:
                # We can break early if prod exceeds some limit? No, products can be large.
                # But we just need to check membership in products_set.
                
                new_prods = []
                prod = 1
                valid = True
                for i in range(len(seq) - 1, -1, -1):
                    prod *= seq[i]
                    if prod in products_set:
                        valid = False
                        break
                    new_prods.append(prod)
                
                # Also check x itself
                if x in products_set:
                    valid = False
                else:
                    new_prods.append(x)
                
                if not valid:
                    continue
                
                # Check if any new_prod is in products_set
                # We already checked prod in products_set in the loop above?
                # Wait, the loop checks `prod` (which is product of seq[i:] * x? No)
                # Let's re-evaluate.
                # Block is seq[i]...seq[j-1] * x.
                # If we iterate i from len(seq)-1 down to 0:
                # i = len(seq)-1: prod = seq[-1] * x. Check if in products_set.
                # i = len(seq)-2: prod = seq[-2] * seq[-1] * x. Check if in products_set.
                # ...
                # i = 0: prod = seq[0]...seq[-1] * x.
                # Also check x itself (block of length 1).
                
                # If all checks pass, we can recurse.
                
                # We need to add these new products to the set for the next level.
                # To avoid copying the whole set, we can use a list of added items and remove them on backtrack.
                # Or just copy the set. Set size is O(L^2). L ~ 40. Size ~ 1600.
                # Copying 1600 ints is fast.
                
                new_set = set(products_set)
                for p in new_prods:
                    if p in new_set: # Should not happen if valid check passed
                        valid = False
                        break
                    new_set.add(p)
                
                if not valid:
                    continue
                
                dfs(seq + [x], new_set, x)
                
                # Backtrack: nothing to do if we passed new_set by value
        
        dfs([], set(), 1)
        
        return best_seq
