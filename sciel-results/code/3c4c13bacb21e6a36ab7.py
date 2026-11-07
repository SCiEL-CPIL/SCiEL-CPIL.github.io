import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Prime factorization helper for faster collision checking
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    # Precompute factorizations for all candidates
    factorizations = {n: factorize(n) for n in candidates}
    
    # 1. Direct Construction using primes (guaranteed distinct products)
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    # Size: 12
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    # Helper: check if adding x to current_set keeps subset products distinct
    # Using exponent vectors: represent each product as a tuple of exponents
    # Two products collide iff their exponent vectors are equal
    def check_valid_addition(current_set, x, factorizations):
        # Build exponent vector for x
        x_vec = factorizations[x]
        
        # For each existing product (represented by its exponent vector),
        # check if adding x_vec creates a collision
        # We need to track all 2^k subset products as exponent vectors
        
        # Start with the empty product (zero vector)
        products = [{}]  # list of exponent dicts
        
        for elem in current_set:
            elem_vec = factorizations[elem]
            new_products = []
            for prod in products:
                new_prod = dict(prod)
                for p, exp in elem_vec.items():
                    new_prod[p] = new_prod.get(p, 0) + exp
                new_products.append(new_prod)
            products.extend(new_products)
        
        # Now check if adding x creates any collision
        x_vec = factorizations[x]
        existing = set()
        for prod in products:
            # Convert to tuple for hashing
            key = tuple(sorted(prod.items()))
            existing.add(key)
        
        for prod in products:
            new_prod = dict(prod)
            for p, exp in x_vec.items():
                new_prod[p] = new_prod.get(p, 0) + exp
            key = tuple(sorted(new_prod.items()))
            if key in existing:
                return False
            existing.add(key)
        
        return True
    
    # Faster check using direct product computation (for small k)
    def check_valid_direct(current_set, x):
        products = {1}
        for elem in current_set:
            new_prods = {p * elem for p in products}
            products.update(new_prods)
        
        new_prods = {p * x for p in products}
        if new_prods & products:
            return False
        return True
    
    random.seed(42)
    
    # 2. Randomized greedy search with multiple restarts
    # Use direct product check for speed when k is small, exponent vectors when k is large
    for restart in range(2000):
        if time.time() > deadline - 0.5:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_products = {1}
        
        for x in shuffled:
            # Choose check method based on current size
            if len(current_set) <= 15:
                valid = check_valid_direct(current_set, x)
            else:
                valid = check_valid_addition(current_set, x, factorizations)
            
            if valid:
                current_set.append(x)
                if len(current_set) <= 15:
                    new_prods = {p * x for p in current_products}
                    current_products.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 16:
                # Good improvement, continue searching but with less time
                pass
    
    # 3. Local search: try to improve best_set by swaps and additions
    if time.time() < deadline - 5:
        local_best = best_set[:]
        local_best_size = best_size
        local_products = {1}
        for x in local_best:
            new_prods = {p * x for p in local_products}
            local_products.update(new_prods)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            # Try adding each candidate not in set
            for x in candidates:
                if x in local_best:
                    continue
                if check_valid_direct(local_best, x):
                    local_best.append(x)
                    new_prods = {p * x for p in local_products}
                    local_products.update(new_prods)
                    improved = True
                    if len(local_best) > local_best_size:
                        local_best_size = len(local_best)
                        best_set = local_best[:]
                        best_size = local_best_size
                    break  # Restart addition loop after one success
            
            # Try swaps: remove one, add another
            if not improved:
                for i, old_x in enumerate(local_best[:]):
                    # Remove old_x
                    trial_set = [x for j, x in enumerate(local_best) if j != i]
                    trial_products = {1}
                    for elem in trial_set:
                        new_prods = {p * elem for p in trial_products}
                        trial_products.update(new_prods)
                    
                    for new_x in candidates:
                        if new_x in trial_set:
                            continue
                        if check_valid_direct(trial_set, new_x):
                            # Accept swap
                            local_best = trial_set + [new_x]
                            local_products = {1}
                            for elem in local_best:
                                new_prods = {p * elem for p in local_products}
                                local_products.update(new_prods)
                            improved = True
                            if len(local_best) > local_best_size:
                                local_best_size = len(local_best)
                                best_set = local_best[:]
                                best_size = local_best_size
                            break
                    if improved:
                        break
    
    # 4. More targeted search: sort by different heuristics
    if time.time() < deadline - 10:
        # Try sorting by number of prime factors (fewer = safer)
        def num_prime_factors(n):
            count = 0
            d = 2
            while d * d <= n:
                while n % d == 0:
                    count += 1
                    n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        
        sorted_by_pf = sorted(candidates, key=lambda x: num_prime_factors(x))
        current_set = []
        current_products = {1}
        for x in sorted_by_pf:
            if check_valid_direct(current_set, x):
                current_set.append(x)
                new_prods = {p * x for p in current_products}
                current_products.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # Try sorting by value (ascending)
        current_set = []
        current_products = {1}
        for x in sorted(candidates):
            if check_valid_direct(current_set, x):
                current_set.append(x)
                new_prods = {p * x for p in current_products}
                current_products.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # Try sorting by value (descending)
        current_set = []
        current_products = {1}
        for x in sorted(candidates, reverse=True):
            if check_valid_direct(current_set, x):
                current_set.append(x)
                new_prods = {p * x for p in current_products}
                current_products.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # 5. Final randomized restarts with time remaining
    start_time = time.time()
    max_time = max(0, deadline - time.time() - 1.0)
    restarts = 0
    while time.time() - start_time < max_time:
        restarts += 1
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_products = {1}
        
        for x in shuffled:
            if len(current_set) <= 15:
                valid = check_valid_direct(current_set, x)
            else:
                valid = check_valid_addition(current_set, x, factorizations)
            
            if valid:
                current_set.append(x)
                if len(current_set) <= 15:
                    new_prods = {p * x for p in current_products}
                    current_products.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Final check for valid types
    return [int(x) for x in best_set]
