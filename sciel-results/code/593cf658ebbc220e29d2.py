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

    def is_valid_fast(seq):
        """
        Optimized validity check using early termination and product bounding.
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

    def check_insertion(S, x):
        """
        Check if inserting x into sorted S at its sorted position keeps validity.
        Only checks products involving x against all existing products.
        """
        n = len(S)
        idx = bisect.bisect_left(S, x)
        
        # First, collect all existing products
        existing_products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= S[j]
                existing_products.add(prod)
        
        # Now check all products involving x in the new sequence
        new_S = S[:idx] + [x] + S[idx:]
        new_n = n + 1
        
        for i in range(new_n):
            if new_S[i] != x and i < idx:
                # Product doesn't involve x, skip (already in existing_products)
                # But wait - products that span across the insertion point DO change
                # Actually, products not involving x are the same multiset as before
                # But products that include x need checking
                pass
            if i >= idx:
                # This product starts at or after x's position
                # Check if it involves x
                if i == idx:
                    # Starts at x
                    prod = 1
                    for j in range(i, new_n):
                        prod *= new_S[j]
                        if prod in existing_products:
                            return False
                        existing_products.add(prod)
                else:
                    # Starts after x, doesn't involve x - skip
                    pass
            elif i < idx:
                # Starts before x, check if it extends to include x
                prod = 1
                for j in range(i, new_n):
                    prod *= new_S[j]
                    if j >= idx:
                        # This product involves x
                        if prod in existing_products:
                            return False
                        existing_products.add(prod)
                    else:
                        # Doesn't involve x yet, but might later
                        pass
        
        # Also need to check products that start before x and end after x
        # These were not fully checked above in all cases
        # Let's do a cleaner check
        
        # Clean approach: collect all products in new_S, check for duplicates
        # But only check products that involve x against existing, plus check x-products among themselves
        all_new_products = set()
        for i in range(new_n):
            prod = 1
            for j in range(i, new_n):
                prod *= new_S[j]
                if prod in all_new_products:
                    return False
                all_new_products.add(prod)
        
        return True

    # 1. Greedy Initialization with randomization
    best_S = []
    best_len = 0
    
    for restart in range(5):
        if time.time() - start_time > limit:
            break
            
        S = []
        if restart == 0:
            # Deterministic greedy
            for x in range(2, 129):
                if is_valid_fast(S + [x]):
                    S.append(x)
        else:
            # Randomized greedy: shuffle candidates at each step
            remaining = list(range(2, 129))
            random.shuffle(remaining)
            for x in remaining:
                if is_valid_fast(S + [x]):
                    S.append(x)
            S.sort()
        
        if len(S) > best_len:
            best_S = S[:]
            best_len = len(S)
    
    S = best_S[:]
    
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
            
            if is_valid_fast(new_S):
                S = new_S
                improved = True
                if len(S) > best_len:
                    best_S = S[:]
                    best_len = len(S)
                break
        
        if time.time() - start_time > limit:
            break
            
        if not improved:
            # Step B: Try 1-out, 2-in moves to increase length by 1
            if len(S) < 127:
                moves_tried = 0
                max_moves = 100
                while moves_tried < max_moves and time.time() - start_time < limit:
                    moves_tried += 1
                    
                    # Pick a random element to remove
                    y = random.choice(S)
                    candidates = [x for x in range(2, 129) if x not in S and x != y]
                    if len(candidates) < 2:
                        continue
                    
                    # Try random pairs with some structure
                    found = False
                    for _ in range(20):
                        if time.time() - start_time > limit:
                            break
                        x1, x2 = random.sample(candidates, 2)
                        if x1 > x2:
                            x1, x2 = x2, x1
                        
                        new_S = [v for v in S if v != y]
                        new_S.append(x1)
                        new_S.append(x2)
                        new_S.sort()
                        
                        if is_valid_fast(new_S):
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
            
            # Step C: Try 2-out, 2-in moves (same length, different composition)
            if len(S) >= 4:
                swap_tried = 0
                while swap_tried < 40 and time.time() - start_time < limit:
                    swap_tried += 1
                    
                    # Pick two elements to remove
                    y1, y2 = random.sample(S, 2)
                    candidates = [x for x in range(2, 129) if x not in S and x != y1 and x != y2]
                    if len(candidates) < 2:
                        continue
                    
                    for _ in range(15):
                        if time.time() - start_time > limit:
                            break
                        x1, x2 = random.sample(candidates, 2)
                        if x1 > x2:
                            x1, x2 = x2, x1
                        
                        new_S = [v for v in S if v != y1 and v != y2]
                        new_S.append(x1)
                        new_S.append(x2)
                        new_S.sort()
                        
                        if is_valid_fast(new_S):
                            S = new_S
                            # Same length, but may enable future inserts
                            break
                
                if time.time() - start_time > limit:
                    break
            
            # Step D: Try 1-out, 1-in swap to explore
            if len(S) >= 2:
                swap_tried = 0
                while swap_tried < 25 and time.time() - start_time < limit:
                    swap_tried += 1
                    
                    y = random.choice(S)
                    candidates = [x for x in range(2, 129) if x not in S and x != y]
                    if not candidates:
                        continue
                    
                    x = random.choice(candidates)
                    new_S = [v for v in S if v != y]
                    new_S.append(x)
                    new_S.sort()
                    
                    if is_valid_fast(new_S):
                        S = new_S
                        break

    return best_S
