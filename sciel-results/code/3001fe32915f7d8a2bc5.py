import time
import random
from math import isqrt

def _is_prime(a):
    if a < 2: return False
    if a < 4: return True
    if a % 2 == 0 or a % 3 == 0: return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0: return False
        d += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    candidates = list(range(2, 41))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    best_size = 0
    best_set = []
    
    def all_subset_products(S):
        products = {1}
        for x in S:
            new_products = set()
            for p in products:
                new_products.add(p * x)
            products = products.union(new_products)
        return products
    
    def can_add(c, S, products):
        # products is a set of all subset products of S
        new_products = set()
        for p in products:
            new_products.add(p * c)
        # Check intersection
        if products.isdisjoint(new_products):
            return True, products.union(new_products)
        return False, None

    def check_set(set_vals):
        prods = {1}
        for v in set_vals:
            new_prods = list(prods)
            for prod in new_prods:
                if v * prod in prods:
                    return False
                prods.add(v * prod)
        return True

    # Strategy 1: Greedy from primes with multiple restarts
    random.seed(42)
    
    # Initial deterministic attempt
    S = primes.copy()
    products = {1}
    for p in S:
        new_p = set()
        for x in products:
            new_p.add(x * p)
        products = products.union(new_p)
    
    c_list = composites.copy()
    c_list.sort()
    for c in c_list:
        ok, new_products = can_add(c, S, products)
        if ok:
            S.append(c)
            products = new_products
    if len(S) > best_size:
        best_size = len(S)
        best_set = S.copy()
    
    # Randomized restarts
    for _ in range(5000):
        if time.time() > deadline:
            break
        S = primes.copy()
        products = {1}
        for p in S:
            new_p = set()
            for x in products:
                new_p.add(x * p)
            products = products.union(new_p)
        
        c_list = composites.copy()
        random.shuffle(c_list)
        
        for c in c_list:
            ok, new_products = can_add(c, S, products)
            if ok:
                S.append(c)
                products = new_products
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            if best_size >= 18:
                return best_set
    
    # Local search: try to improve by swapping elements
    for trial in range(200):
        if time.time() > deadline:
            break
        current = best_set.copy()
        current_prods = all_subset_products(current)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # Try removing one element and adding two others
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                s_val = current[i]
                without_i = [x for j, x in enumerate(current) if j != i]
                
                # Recompute products for without_i
                prods_without = all_subset_products(without_i)
                
                # Try adding two composites not in current
                remaining = [x for x in composites if x not in current]
                random.shuffle(remaining)
                
                for c1 in remaining[:10]:
                    ok1, new_prods1 = can_add(c1, without_i, prods_without)
                    if not ok1:
                        continue
                    for c2 in remaining[:10]:
                        if c2 == c1:
                            continue
                        ok2, new_prods2 = can_add(c2, without_i, new_prods1)
                        if ok2:
                            # Check if c2 also doesn't conflict with c1's products
                            # Actually, we need to check if the set {without_i, c1, c2} is valid
                            candidate = without_i + [c1, c2]
                            if check_set(candidate):
                                current = candidate
                                current_prods = all_subset_products(current)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = current.copy()
                if best_size >= 18:
                    return best_set
    
    # Final local search: try single swaps
    for trial in range(100):
        if time.time() > deadline:
            break
        current = best_set.copy()
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                s_val = current[i]
                without_i = [x for j, x in enumerate(current) if j != i]
                prods_without = all_subset_products(without_i)
                
                remaining = [x for x in composites if x not in current]
                random.shuffle(remaining)
                
                for c in remaining[:20]:
                    ok, new_prods = can_add(c, without_i, prods_without)
                    if ok:
                        candidate = without_i + [c]
                        if check_set(candidate):
                            current = candidate
                            improved = True
                            break
                if improved:
                    break
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = current.copy()
                if best_size >= 18:
                    return best_set
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_set]
