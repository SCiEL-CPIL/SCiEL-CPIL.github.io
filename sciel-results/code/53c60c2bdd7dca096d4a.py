"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    def greedy():
        """Standard greedy: add x if no product collision."""
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

    def get_products(seq):
        """Compute all non-empty consecutive block products for a sequence."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        """Check if all non-empty consecutive block products are distinct."""
        prods = get_products(seq)
        # Number of non-empty consecutive blocks is n*(n+1)/2
        n = len(seq)
        expected = n * (n + 1) // 2
        return len(prods) == expected

    def local_search(seq, time_limit):
        """
        Iterative swap-and-insert local search.
        Try to improve length by:
        1. Removing one element and trying to insert a different one.
        2. Swapping one element with a candidate.
        3. Inserting a new element at any position.
        """
        best = list(seq)
        best_len = len(best)
        
        # Precompute products for best
        best_prods = get_products(best)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Strategy 1: Try to insert a new element at any position
            # For each position i (0 to len(best)), try each candidate x not in best
            # such that best[i-1] < x < best[i] (with boundaries)
            for pos in range(len(best) + 1):
                if time.time() > time_limit:
                    break
                
                # Determine valid range for x at position pos
                lo = best[pos - 1] + 1 if pos > 0 else 2
                hi = best[pos] - 1 if pos < len(best) else 128
                
                if lo > hi:
                    continue
                    
                for x in range(lo, hi + 1):
                    if time.time() > time_limit:
                        break
                    
                    # Check if x is already in best
                    if x in best:
                        continue
                        
                    new_seq = best[:pos] + [x] + best[pos:]
                    
                    # Check validity
                    new_prods = get_products(new_seq)
                    n = len(new_seq)
                    expected = n * (n + 1) // 2
                    if len(new_prods) != expected:
                        continue
                        
                    # Valid insertion!
                    best = new_seq
                    best_prods = new_prods
                    best_len = len(best)
                    improved = True
                    break  # Restart loop with new best
            
            if not improved:
                # Strategy 2: Try swap-and-insert
                # Remove one element at position i, then try to insert a different element
                for i in range(len(best)):
                    if time.time() > time_limit:
                        break
                        
                    removed = best[i]
                    temp_seq = best[:i] + best[i+1:]
                    temp_prods = get_products(temp_seq)
                    
                    # Now try to insert a new element x into temp_seq
                    # x must be > temp_seq[i-1] (or >= 2) and < temp_seq[i] (or <= 128)
                    # But we also want to potentially reinsert 'removed' or a new value
                    # Let's try inserting any x that makes the sequence valid and longer than current best? 
                    # No, temp_seq is shorter by 1. We want to end up with same length but different composition,
                    # or potentially allow future insertions.
                    # Actually, the goal of swap is to change the set of products to allow future insertions.
                    # So we accept a swap if it results in a valid sequence of the same length.
                    
                    for pos in range(len(temp_seq) + 1):
                        if time.time() > time_limit:
                            break
                            
                        lo = temp_seq[pos - 1] + 1 if pos > 0 else 2
                        hi = temp_seq[pos] - 1 if pos < len(temp_seq) else 128
                        
                        if lo > hi:
                            continue
                            
                        # We want to insert x. If x == removed, we revert.
                        # If x != removed, we have a swap.
                        for x in range(lo, hi + 1):
                            if x == removed:
                                continue
                                
                            new_seq = temp_seq[:pos] + [x] + temp_seq[pos:]
                            # Check strictly increasing
                            # Since we picked x in range, it is increasing.
                            
                            new_prods = get_products(new_seq)
                            n = len(new_seq)
                            expected = n * (n + 1) // 2
                            if len(new_prods) != expected:
                                continue
                                
                            # Valid swap. Accept it to explore new space.
                            # We don't necessarily gain length here, but we change the product set.
                            best = new_seq
                            best_prods = new_prods
                            # best_len stays the same
                            improved = True
                            break
                        
                        if improved:
                            break
                    if improved:
                        break
                        
    # 1. Start with greedy
    greedy_seq, _ = greedy()
    current_best = greedy_seq
    best_len = len(current_best)
    
    # 2. Apply local search
    if time.time() < deadline - 5:
        current_best = local_search(current_best, deadline - 5)
    
    # 3. Optional: If we have time, try a different greedy order or random restarts?
    # For now, just return the best found.
    
    return current_best

# EVOLVE-BLOCK-END
