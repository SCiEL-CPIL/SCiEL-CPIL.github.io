"""New program: greedy + optimized local search for insertion."""
import math

def construct():
    # Phase 1: Greedy initialization
    values = []
    products = set()
    for x in range(2, 129):
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    # Phase 2: Local search - try to insert one element at a time
    n_initial = len(values)
    for iteration in range(50):  # Limit iterations
        inserted = False
        for pos in range(len(values) + 1):
            for x in range(2, 129):
                if x in values:
                    continue
                # Quick check: x must fit in sorted order
                if pos < len(values) and x >= values[pos]:
                    continue
                if pos > 0 and x <= values[pos - 1]:
                    continue
                
                # Build candidate
                candidate = values[:pos] + [x] + values[pos:]
                
                # Check validity efficiently
                valid = True
                prod_set = set()
                n = len(candidate)
                for i in range(n):
                    prod = 1
                    for j in range(i, n):
                        prod *= candidate[j]
                        if prod in prod_set:
                            valid = False
                            break
                        prod_set.add(prod)
                    if not valid:
                        break
                
                if valid:
                    values = candidate
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            break
    
    return values
