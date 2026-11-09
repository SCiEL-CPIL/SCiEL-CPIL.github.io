"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # Initial greedy
    values = []
    products = set()
    for x in range(2, 129):
        tails = []
        prod = x
        for a in reversed(values):
            tails.append(prod)
            prod *= a
        tails.append(x)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best = list(values)
    best_len = len(best)
    
    def try_add_missing(base, base_prods):
        """Try to add missing elements to base in increasing order."""
        improved = list(base)
        imp_prods = set(base_prods)
        for x in range(2, 129):
            if time.time() > deadline:
                break
            if x in improved:
                continue
            insert_pos = len(improved)
            for i, v in enumerate(improved):
                if v > x:
                    insert_pos = i
                    break
            new_seq = improved[:insert_pos] + [x] + improved[insert_pos:]
            # Check all consecutive products in new_seq
            seen = set()
            valid = True
            for i in range(len(new_seq)):
                prod = 1
                for j in range(i, len(new_seq)):
                    prod *= new_seq[j]
                    if prod in seen or prod in imp_prods:
                        valid = False
                        break
                    seen.add(prod)
                if not valid:
                    break
            if valid:
                improved = new_seq
                imp_prods = seen
        return improved
    
    # Try removing each element and re-adding
    for ri in range(len(best)):
        if time.time() > deadline:
            break
        candidate = best[:ri] + best[ri+1:]
        # Build product set
        cand_prods = set()
        n = len(candidate)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= candidate[j]
                cand_prods.add(prod)
        
        improved = try_add_missing(candidate, cand_prods)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)
    
    return best
# EVOLVE-BLOCK-END
