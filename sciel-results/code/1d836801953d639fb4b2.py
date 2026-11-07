import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Prime factorization helpers
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def factor_vec(n):
        v = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        return tuple(v)

    # Precompute exponent vectors for 2..40
    vecs = {}
    for n in range(2, 41):
        vecs[n] = factor_vec(n)

    # Subset product validity check
    # A set of numbers has distinct subset products if and only if
    # all 2^k subset products are distinct.
    # We can check this incrementally: when adding a number x to a set S,
    # the new subset products are old_products and old_products * x.
    # These must be disjoint from old_products.

    def can_add(products_set, x):
        """Check if x can be added to the set represented by products_set."""
        # If any p * x is already in products_set, we have a collision
        for p in products_set:
            if p * x in products_set:
                return False
        return True

    def add_to_products(products_set, x):
        """Add x to the set, returning the new products set."""
        new_products = set(products_set)
        for p in products_set:
            new_products.add(p * x)
        return new_products

    # Greedy construction with a given order
    def greedy_construct(order):
        products = {1}
        selected = []
        for n in order:
            if can_add(products, n):
                selected.append(n)
                products = add_to_products(products, n)
        return selected, products

    # Local search: try swapping elements to increase set size
    def local_search(initial_set, initial_products, max_iter=500):
        best_set = list(initial_set)
        best_products = initial_products
        best_size = len(best_set)
        
        if best_size >= 18:
            return best_set, best_products
        
        rng = random.Random(42)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Try single swaps
            improved = False
            current_set = list(best_set)
            current_set_set = set(current_set)
            
            # Try removing one element and adding another
            for i in range(len(current_set)):
                if time.time() > deadline - 5:
                    break
                    
                removed = current_set[i]
                # Recompute products without removed
                products_without = {1}
                for x in current_set:
                    if x == removed:
                        continue
                    products_without = add_to_products(products_without, x)
                
                # Try adding candidates not in current set
                for candidate in range(2, 41):
                    if candidate in current_set_set:
                        continue
                    
                    if can_add(products_without, candidate):
                        new_set = current_set[:i] + current_set[i+1:] + [candidate]
                        new_products = add_to_products(products_without, candidate)
                        
                        if len(new_set) > best_size:
                            best_set = new_set
                            best_products = new_products
                            best_size = len(new_set)
                            improved = True
                            if best_size >= 18:
                                return best_set, best_products
                            break
            
            # Try adding without removing (if possible)
            if not improved:
                for candidate in range(2, 41):
                    if candidate in set(best_set):
                        continue
                    if can_add(best_products, candidate):
                        best_set.append(candidate)
                        best_products = add_to_products(best_products, candidate)
                        best_size += 1
                        improved = True
                        if best_size >= 18:
                            return best_set, best_products
                        break
            
            if not improved:
                break
        
        return best_set, best_products

    # Try multiple greedy orders
    rng = random.Random(12345)
    
    # Initial set from seed program
    initial_guess = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify initial guess
    prods = {1}
    valid = True
    for x in initial_guess:
        if not can_add(prods, x):
            valid = False
            break
        prods = add_to_products(prods, x)
    
    if valid:
        best_set = initial_guess
        best_size = len(best_set)
    else:
        best_set = [2]
        best_size = 1
        prods = {1, 2}

    # Local search refinement
    if best_size < 18:
        sel, prods = local_search(best_set, prods, max_iter=1000)
        if len(sel) > best_size:
            best_set = sel
            best_size = len(best_set)
            prods = prods

    # Multiple random restarts
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        if best_size >= 18:
            break
        
        order = list(range(2, 41))
        rng.shuffle(order)
        
        sel, prods = greedy_construct(order)
        
        if len(sel) > best_size:
            best_set = sel
            best_size = len(sel)
            sel2, prods2 = local_search(sel, prods, max_iter=500)
            if len(sel2) > best_size:
                best_set = sel2
                best_size = len(best_set)
                prods = prods2
        elif len(sel) >= best_size - 2 and len(sel) < 18:
            sel2, prods2 = local_search(sel, prods, max_iter=200)
            if len(sel2) > best_size:
                best_set = sel2
                best_size = len(best_set)
                prods = prods2

    # Final verification
    if best_size > 0:
        prods = {1}
        valid = True
        for x in best_set:
            if not can_add(prods, x):
                valid = False
                break
            prods = add_to_products(prods, x)
        
        if not valid:
            return [2]

    return best_set
