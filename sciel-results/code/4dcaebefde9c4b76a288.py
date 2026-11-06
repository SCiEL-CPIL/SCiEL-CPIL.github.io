"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def construct():
    def standard_greedy():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = standard_greedy()
    best_len = len(best_seq)
    
    start_time = time.time()
    time_limit = 55.0
    
    sys.setrecursionlimit(200)
    
    def dfs(idx, values, products):
        nonlocal best_len, best_seq
        nonlocal start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        remaining = 128 - idx
        if len(values) + remaining <= best_len:
            return
            
        if idx > 128:
            if len(values) > best_len:
                best_len = len(values)
                best_seq = list(values)
            return
        
        x = idx
        
        # Option 1: Include x
        tails = [x]
        prod = x
        valid = True
        for a in reversed(values):
            prod *= a
            tails.append(prod)
            if prod in products:
                valid = False
                break
        
        if valid and x not in products:
            # Check other tails
            conflict = False
            for t in tails[1:]:
                if t in products:
                    conflict = True
                    break
            if not conflict:
                values.append(x)
                products.update(tails)
                dfs(idx + 1, values, products)
                values.pop()
                products.difference_update(tails)
        
        # Option 2: Exclude x
        dfs(idx + 1, values, products)
        
    dfs(2, [], set())
    
    return best_seq

# EVOLVE-BLOCK-END
