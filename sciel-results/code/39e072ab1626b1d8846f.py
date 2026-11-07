import time
import random
from math import gcd
from collections import Counter

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Precompute prime factorizations as exponent vectors
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def factorize(n):
        """Return tuple of exponents for primes list."""
        exponents = []
        for p in primes:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            exponents.append(count)
            if n == 1:
                break
        # If n > 1, it's a prime > 37 (but our range is 2-40, so max prime is 37)
        # Actually all primes in 2-40 are in our list, so n should be 1
        return tuple(exponents)
    
    # Precompute exponent vectors for all candidates
    exp_vectors = {n: factorize(n) for n in candidates}
    
    # Direct construction using primes (guaranteed distinct products)
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    def build_exp_products(current_set):
        """Build the set of all subset exponent vectors for current_set."""
        products = {tuple([0]*len(primes))}
        for x in current_set:
            ex = exp_vectors[x]
            new_prods = set()
            for p in products:
                new_prods.add(tuple(p[i] + ex[i] for i in range(len(primes))))
            products.update(new_prods)
        return products
    
    def is_valid_addition_exp(current_exp_products, candidate):
        """Check if adding candidate keeps all exponent vectors distinct."""
        ex = exp_vectors[candidate]
        for p in current_exp_products:
            new_p = tuple(p[i] + ex[i] for i in range(len(primes)))
            if new_p in current_exp_products:
                return False
        return True
    
    def try_add_to_set_exp(current_set, current_exp_products, candidate):
        """Try to add candidate using exponent vectors."""
        if is_valid_addition_exp(current_exp_products, candidate):
            ex = exp_vectors[candidate]
            new_prods = set()
            for p in current_exp_products:
                new_prods.add(tuple(p[i] + ex[i] for i in range(len(primes))))
            new_exp_products = current_exp_products | new_prods
            new_set = current_set + [candidate]
            return (True, new_set, new_exp_products)
        return (False, None, None)
    
    def greedy_from_order_exp(ordered_candidates):
        """Greedy construction from a given ordering using exponent vectors."""
        current_set = []
        current_exp_products = {tuple([0]*len(primes))}
        for x in ordered_candidates:
            success, new_set, new_exp_products = try_add_to_set_exp(current_set, current_exp_products, x)
            if success:
                current_set = new_set
                current_exp_products = new_exp_products
        return current_set
    
    def local_search_add_exp(current_set):
        """Try to add more elements via local search using exponent vectors."""
        current_exp_products = build_exp_products(current_set)
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for x in candidates:
                if x in current_set:
                    continue
                success, new_set, new_exp_products = try_add_to_set_exp(current_set, current_exp_products, x)
                if success:
                    current_set = new_set
                    current_exp_products = new_exp_products
                    added = True
                    break
        return current_set
    
    def local_search_swap_exp(current_set):
        """Try to improve by swapping one element for another."""
        current_exp_products = build_exp_products(current_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i, elem in enumerate(current_set):
                remaining = current_set[:i] + current_set[i+1:]
                remaining_exp_products = build_exp_products(remaining)
                for x in candidates:
                    if x in remaining:
                        continue
                    success, new_set, new_exp_products = try_add_to_set_exp(remaining, remaining_exp_products, x)
                    if success:
                        current_set = new_set
                        current_exp_products = new_exp_products
                        improved = True
                        break
                if improved:
                    break
        return current_set
    
    random.seed(42)
    
    # Phase 1: Randomized greedy with multiple restarts
    for restart in range(200):
        if time.time() > deadline - 3:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = greedy_from_order_exp(shuffled)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 17:
                break
    
    # Phase 2: Local search - try to add elements
    if time.time() < deadline - 4:
        current_set = local_search_add_exp(best_set[:])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 3: Heuristic orderings
    if time.time() < deadline - 6:
        sort_keys = [
            lambda x: x,
            lambda x: -x,
            lambda x: sum(exp_vectors[x]),
            lambda x: -sum(exp_vectors[x]),
            lambda x: max(exp_vectors[x], default=0),
            lambda x: sum(1 for e in exp_vectors[x] if e > 0),
        ]
        for sort_key in sort_keys:
            if time.time() > deadline - 3:
                break
            
            sorted_candidates = sorted(candidates, key=sort_key)
            current_set = greedy_from_order_exp(sorted_candidates)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
    
    # Phase 4: Swap-based local search
    if time.time() < deadline - 5:
        current_set = local_search_swap_exp(best_set[:])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 5: Final random restarts with polish
    while time.time() < deadline - 1:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = greedy_from_order_exp(shuffled)
        
        if time.time() < deadline - 1:
            current_set = local_search_add_exp(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Final verification: ensure all subset products are truly distinct
    def verify_distinct_products(s):
        """Verify all 2^k subset products are distinct."""
        if not s:
            return True
        products = {1}
        for x in s:
            new_prods = set()
            for p in products:
                new_prods.add(p * x)
            if len(products | new_prods) != len(products) + len(new_prods):
                return False
            products.update(new_prods)
        return True
    
    # Verify best_set, if invalid, fall back to primes
    if not verify_distinct_products(best_set):
        best_set = direct_primes[:]
        best_size = len(best_set)
    
    return [int(x) for x in best_set]
