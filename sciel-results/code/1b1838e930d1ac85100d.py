import random
import time

def construct():
    # Start with the greedy solution
    def greedy(start_val, current_values, current_products):
        values = list(current_values)
        products = set(current_products)
        last = values[-1] if values else 1
        for x in range(max(2, last + 1), 129):
            # Compute new tails
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
                # Early termination if prod is huge? 
                # Python ints are fine, but checking intersection is O(N)
            
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values, products

    # Initial greedy
    best_values, best_products = greedy(2, [], set())
    best_len = len(best_values)
    
    # Local Search / Restart Strategy
    # We try to improve by removing some elements and re-greedy-filling
    start_time = time.time()
    iterations = 0
    
    # We will try to remove 1 or 2 random elements from the best solution
    # and then re-run greedy from that point to see if we get a longer sequence
    
    # To make it efficient, we only do this if we have time
    while time.time() - start_time < 50:
        iterations += 1
        # Pick a random subset of elements to remove
        # Usually removing 1 element is enough to unlock a different path
        if random.random() < 0.5:
            remove_count = 1
        else:
            remove_count = 2
            
        if best_len <= remove_count:
            continue
            
        # Choose random indices to remove
        indices_to_remove = set()
        while len(indices_to_remove) < remove_count:
            idx = random.randint(0, best_len - 1)
            indices_to_remove.add(idx)
            
        # Create a new sequence without these elements
        new_values = [v for i, v in enumerate(best_values) if i not in indices_to_remove]
        
        # We need to recompute products for the new prefix
        # Since the sequence is increasing, we can just re-greedy from the last kept element
        # But wait, the products of the *prefix* might be different if we removed an element in the middle?
        # No, the sequence is strictly increasing. If we remove an element, the remaining elements keep their relative order.
        # However, the "consecutive block" property changes.
        # E.g., [2, 3, 4] -> remove 3 -> [2, 4].
        # Products of [2,3,4]: 2,3,4,6,12,24.
        # Products of [2,4]: 2,4,8.
        # The set of products is different.
        # So we cannot just reuse the old products. We must recompute or re-greedy.
        
        # Efficient way:
        # Take the prefix up to the first removed element? No.
        # Just re-run greedy from the beginning? Too slow if done many times.
        
        # Better:
        # We know the prefix before the first removed element is valid and its products are known.
        # But implementing this incrementally is complex.
        
        # Simpler: Just re-run greedy from the start with a "bias" to include the removed elements?
        # Or just re-run greedy on the remaining candidates?
        
        # Let's try a different approach:
        # Randomized Greedy.
        # At each step, pick a random valid candidate from the next K candidates.
        
        # Let's implement a simple Randomized Greedy
        values = []
        products = set()
        
        for _ in range(127): # Max possible steps
            # Find all valid candidates > last
            last = values[-1] if values else 1
            candidates = []
            for x in range(last + 1, 129):
                tails = [x]
                prod = x
                for a in reversed(values):
                    prod *= a
                    tails.append(prod)
                
                if not products.intersection(tails):
                    candidates.append(x)
            
            if not candidates:
                break
            
            # Pick one randomly
            # Bias towards smaller numbers? Or uniform?
            # Smaller numbers create more conflicts later, but are more "dense".
            # Let's pick uniformly.
            x = random.choice(candidates)
            
            # Add it
            values.append(x)
            # Update products
            tails = [x]
            prod = x
            for a in reversed(values[:-1]): # Exclude x itself from the loop for prod calc? 
                # Wait, the loop above calculated tails for the candidate x given the OLD values.
                # Now we add x to values. The new products are the tails we just computed.
                pass
            
            # We need to re-calculate tails for the set intersection update?
            # No, the `tails` computed in the candidate check are exactly the new products added.
            products.update(tails)
            
        if len(values) > best_len:
            best_len = len(values)
            best_values = values
            best_products = products
            
    return best_values
