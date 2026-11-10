import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    all_numbers = list(range(2, 129))
    best_seq = []
    best_len = 0
    
    def check_validity(seq):
        """Check if a sequence has distinct consecutive products."""
        products = set()
        for i in range(len(seq)):
            prod = 1
            for j in range(i, len(seq)):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def get_products(seq):
        products = set()
        for i in range(len(seq)):
            prod = 1
            for j in range(i, len(seq)):
                prod *= seq[j]
                products.add(prod)
        return products

    # 1. Randomized Greedy
    for _ in range(20000): # Adjust based on speed
        if time.time() > deadline - 10.0: # Leave time for local search
            break
        random_order = all_numbers[:]
        random.shuffle(random_order)
        
        seq = []
        products = set()
        for x in random_order:
            new_prods = [x]
            prod = x
            for a in reversed(seq):
                prod *= a
                new_prods.append(prod)
            if not products.intersection(new_prods):
                seq.append(x)
                products.update(new_prods)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    # 2. Local Search (Insertion)
    # Try to insert unused numbers into the best sequence
    # This is O(N^2 * CheckCost) per iteration
    # We do a few passes
    
    used_in_best = set(best_seq)
    unused = [x for x in all_numbers if x not in used_in_best]
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Shuffle unused to randomize order
        random.shuffle(unused)
        for x in unused:
            # Try inserting x at each position
            inserted = False
            for i in range(len(best_seq) + 1):
                new_seq = best_seq[:i] + [x] + best_seq[i:]
                # Quick check: is it valid?
                # We can optimize this check, but for N<50 it's fine.
                if check_validity(new_seq):
                    best_seq = new_seq
                    best_len += 1
                    used_in_best.add(x)
                    unused.remove(x)
                    improved = True
                    inserted = True
                    break # Try next unused number
            if time.time() > deadline - 2.0:
                break

    return best_seq
