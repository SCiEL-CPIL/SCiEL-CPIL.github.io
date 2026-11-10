import time
import random
import bisect

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin
    
    def is_valid(seq):
        """
        Check if the sequence has distinct products for all non-empty consecutive blocks.
        seq must be sorted.
        """
        n = len(seq)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def is_valid_insertion(S, x):
        """
        Check if inserting x into sorted S at its sorted position keeps validity.
        Only checks products involving x and their interactions with existing products.
        """
        n = len(S)
        idx = bisect.bisect_left(S, x)
        
        # Build the new sequence
        new_S = S[:idx] + [x] + S[idx:]
        new_n = n + 1
        
        # We need to check all products in new_S for duplicates.
        # Optimization: pre-compute products that don't involve x, then check those involving x.
        # Products not involving x are the same as in S (just shifted indices if x inserted before).
        # Actually, products that span across the insertion point change.
        # Simpler: just check all products in new_S.
        # But we can short-circuit: first check if any product in S already equals a product involving x.
        
        # Let's just do the full check but with early termination
        products = set()
        for i in range(new_n):
            prod = 1
            for j in range(i, new_n):
                prod *= new_S[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    # 1. Greedy Initialization
    S = []
    for x in range(2, 129):
        if is_valid_insertion(S, x):
            S.append(x)
    
    best_S = S[:]
    best_len = len(S)
    
    # 2. Local Search / Hill Climbing
    while time.time() - start_time < limit:
        improved = False
        
        # Step A: Try to insert any single missing number
        missing = [x for x in range(2, 129) if x not in S]
        random.shuffle(missing)
        
        for x in missing:
            if time.time() - start_time > limit:
                break
            idx = bisect.bisect_left(S, x)
            new_S = S[:idx] + [x] + S[idx:]
            
            if is_valid(new_S):
                S = new_S
                improved = True
                if len(S) > best_len:
                    best_S = S[:]
                    best_len = len(S)
                break
        
        if time.time() - start_time > limit:
            break
            
        if not improved:
            # Step B: Try swap moves (remove one, add one) - doesn't change length but may enable future inserts
            # Step C: Try 1-out, 2-in moves to increase length by 1
            
            # Try 1-out, 2-in moves
            if len(S) < 127:
                moves_tried = 0
                max_moves = 80
                while moves_tried < max_moves and time.time() - start_time < limit:
                    moves_tried += 1
                    
                    # Pick a random element to remove
                    y = random.choice(S)
                    candidates = [x for x in range(2, 129) if x not in S and x != y]
                    if len(candidates) < 2:
                        continue
                    
                    # Try random pairs with some structure
                    found = False
                    for _ in range(15):
                        if time.time() - start_time > limit:
                            break
                        x1, x2 = random.sample(candidates, 2)
                        if x1 > x2:
                            x1, x2 = x2, x1
                        
                        new_S = [v for v in S if v != y]
                        new_S.append(x1)
                        new_S.append(x2)
                        new_S.sort()
                        
                        if is_valid(new_S):
                            S = new_S
                            improved = True
                            if len(S) > best_len:
                                best_S = S[:]
                                best_len = len(S)
                            found = True
                            break
                    
                    if found:
                        break
                
                # If we improved, go back to trying inserts
                if improved:
                    continue
            
            # Step D: Try 2-out, 2-in moves (same length, different composition)
            if len(S) >= 4:
                swap_tried = 0
                while swap_tried < 30 and time.time() - start_time < limit:
                    swap_tried += 1
                    
                    # Pick two elements to remove
                    y1, y2 = random.sample(S, 2)
                    candidates = [x for x in range(2, 129) if x not in S and x != y1 and x != y2]
                    if len(candidates) < 2:
                        continue
                    
                    for _ in range(10):
                        if time.time() - start_time > limit:
                            break
                        x1, x2 = random.sample(candidates, 2)
                        if x1 > x2:
                            x1, x2 = x2, x1
                        
                        new_S = [v for v in S if v != y1 and v != y2]
                        new_S.append(x1)
                        new_S.append(x2)
                        new_S.sort()
                        
                        if is_valid(new_S):
                            S = new_S
                            # Same length, but may enable future inserts
                            break
                
                if time.time() - start_time > limit:
                    break
            
            # If nothing improved, try swap (1-out, 1-in) to explore
            if len(S) >= 2:
                swap_tried = 0
                while swap_tried < 20 and time.time() - start_time < limit:
                    swap_tried += 1
                    
                    y = random.choice(S)
                    candidates = [x for x in range(2, 129) if x not in S and x != y]
                    if not candidates:
                        continue
                    
                    x = random.choice(candidates)
                    new_S = [v for v in S if v != y]
                    new_S.append(x)
                    new_S.sort()
                    
                    if is_valid(new_S):
                        S = new_S
                        break

    return best_S
