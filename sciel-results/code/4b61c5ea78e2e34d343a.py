"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    candidates = list(range(2, 41))
    best_set = []
    
    def backtrack(start, current_set, current_products_set):
        nonlocal best_set
        
        # Pruning
        if len(current_set) + (len(candidates) - start) <= len(best_set):
            return
            
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(best_set) >= 18:
            return

        for i in range(start, len(candidates)):
            num = candidates[i]
            
            # Check validity
            new_products = set(current_products_set)
            valid = True
            for p in current_products_set:
                new_p = p * num
                if new_p in new_products:
                    valid = False
                    break
                new_products.add(new_p)
                
            if valid:
                backtrack(i + 1, current_set + [num], new_products)
                if len(best_set) >= 18:
                    return

    initial_products = {1}
    backtrack(0, [], initial_products)
    
    return best_set

# EVOLVE-BLOCK-END
