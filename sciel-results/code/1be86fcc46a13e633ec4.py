import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin

    def greedy_solution():
        seq = []
        products = set()
        for x in range(2, 129):
            if x in products:
                continue
            
            # Check if adding x causes any collision
            # New products are x, x*last, x*last*prev, ...
            # If any of these are in `products`, it's a collision.
            
            # Optimization: Check x first (already done)
            # Then check the rest
            valid = True
            prod = x
            new_tails = [x]
            
            for a in reversed(seq):
                prod *= a
                new_tails.append(prod)
                if prod in products:
                    valid = False
                    break
            
            if valid:
                # Ensure no other collisions (though the loop covers all new tails)
                # The loop breaks on first collision. If no break, all new tails are distinct from old products.
                seq.append(x)
                products.update(new_tails)
        
        return seq

    # Get a good baseline with greedy
    best_seq = greedy_solution()
    best_len = len(best_seq)
    
    # Beam Search with larger width and better sorting
    W = 5000
    beam = [([], set())]
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beam = []
        for values, products in beam:
            # Option 1: Skip x
            new_beam.append((values, products))
            
            # Option 2: Include x
            if x in products:
                continue
                
            prod = x
            valid = True
            new_tails = [x]
            
            for a in reversed(values):
                prod *= a
                new_tails.append(prod)
                if prod in products:
                    valid = False
                    break
            
            if valid:
                new_products = set(products)
                new_products.update(new_tails)
                new_values = values + [x]
                new_beam.append((new_values, new_products))
                
        if time.time() > deadline:
            break
            
        if not new_beam:
            break
            
        # Sort by length descending, then by last value ascending
        # Smaller last value allows more future numbers
        new_beam.sort(key=lambda s: (len(s[0]), s[0][-1] if s[0] else 0), reverse=False)
        # Note: We want Longest Length first (Descending), but Smallest Last Value first (Ascending).
        # Python sort is stable and ascending by default.
        # So we sort by (-len, last_val)
        new_beam.sort(key=lambda s: (-len(s[0]), s[0][-1] if s[0] else 0))
        
        # Select top W
        # To maintain diversity, we can take a mix of top W and random samples from a larger pool
        # But for now, top W is a good heuristic.
        # Let's take top W.
        beam = new_beam[:W]
        
        # Track best
        if beam:
            # The first element in the sorted list is the best candidate
            current_best = beam[0]
            if len(current_best[0]) > best_len:
                best_len = len(current_best[0])
                best_seq = current_best[0]
                
    return best_seq if best_seq else [2]
