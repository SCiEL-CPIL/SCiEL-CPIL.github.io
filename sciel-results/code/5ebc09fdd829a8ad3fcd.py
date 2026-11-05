import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    N = 128  # numbers in [2, 128]
    all_candidates = list(range(2, 129))

    # Greedy baseline
    def greedy_construct():
        values, products = [], set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)

    # Beam search with incremental product set tracking
    # Each beam entry: (sequence, product_set, last_value)
    # Product set contains all consecutive block products

    def beam_search(max_width, max_time):
        # Initialize beam with single elements
        beam = []
        for x in all_candidates:
            beam.append(([x], {x}, x))
        
        current_best_len = 0
        current_best_seq = None
        
        while time.time() - start_time < max_time:
            new_beam = []
            for seq, prods, last_val in beam:
                current_len = len(seq)
                if current_len > current_best_len:
                    current_best_len = current_len
                    current_best_seq = seq[:]
                
                # Try to extend with elements > last_val
                for x in range(last_val + 1, 129):
                    # Check if adding x creates any product collision
                    # New products are: x, x*seq[-1], x*seq[-1]*seq[-2], ...
                    new_prods = [x]
                    prod = x
                    for a in reversed(seq):
                        prod *= a
                        new_prods.append(prod)
                    
                    # Check for collision with existing products
                    collision = False
                    for p in new_prods:
                        if p in prods:
                            collision = True
                            break
                    
                    if not collision:
                        new_prods_set = prods.union(new_prods)
                        new_seq = seq + [x]
                        new_beam.append((new_seq, new_prods_set, x))
            
            if not new_beam:
                break
            
            # Sort beam: prioritize longer sequences, then smaller last value
            # This keeps the most "extensible" candidates
            new_beam.sort(key=lambda item: (-len(item[0]), item[2]))
            
            # Keep top max_width entries
            beam = new_beam[:max_width]
            
            # If all beams are same length and we can't extend, stop
            if beam and all(len(b[0]) == len(beam[0][0]) for b in beam):
                # Check if any can be extended further
                can_extend = False
                for seq, prods, last_val in beam:
                    if last_val < 128:
                        can_extend = True
                        break
                if not can_extend:
                    break
        
        if current_best_seq and current_best_len > 0:
            return current_best_seq
        return None

    # Run beam search with increasing width
    for width in [50, 100, 200, 400, 800]:
        if time.time() - start_time > DEADLINE - 10:
            break
        result = beam_search(width, DEADLINE - (time.time() - start_time))
        if result and len(result) > best_len:
            best_seq = result
            best_len = len(result)

    # Final validation
    def is_valid(seq):
        if len(seq) == 0:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return len(products) == n * (n + 1) // 2

    if not is_valid(best_seq):
        best_seq = greedy_construct()

    return best_seq
