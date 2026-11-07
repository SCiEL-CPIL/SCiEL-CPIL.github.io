import time
import random

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Precompute prime factorizations for all candidates
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
    
    prime_factors = {n: factorize(n) for n in candidates}
    
    # Direct construction using primes as initial best
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    def build_products(current_set):
        """Build the set of all subset products for current_set."""
        products = {1}
        for x in current_set:
            new_prods = set()
            for p in products:
                new_prods.add(p * x)
            products.update(new_prods)
        return products
    
    def is_valid_addition(current_products, candidate):
        """Check if adding candidate to the set represented by current_products keeps all products distinct."""
        for p in current_products:
            if p * candidate in current_products:
                return False
        return True
    
    def try_add_to_set(current_set, current_products, candidate):
        """Try to add candidate to current_set. Returns (success, new_set, new_products) or (False, None, None)."""
        if is_valid_addition(current_products, candidate):
            new_set = current_set + [candidate]
            new_prods = set()
            for p in current_products:
                new_prods.add(p * candidate)
            new_products = current_products | new_prods
            return (True, new_set, new_products)
        return (False, None, None)
    
    def greedy_from_order(ordered_candidates):
        """Greedy construction from a given ordering of candidates."""
        current_set = []
        current_products = {1}
        for x in ordered_candidates:
            success, new_set, new_products = try_add_to_set(current_set, current_products, x)
            if success:
                current_set = new_set
                current_products = new_products
        return current_set
    
    def local_search_add(current_set):
        """Try to add more elements to current_set via local search."""
        current_products = build_products(current_set)
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for x in candidates:
                if x in current_set:
                    continue
                success, new_set, new_products = try_add_to_set(current_set, current_products, x)
                if success:
                    current_set = new_set
                    current_products = new_products
                    added = True
                    break
        return current_set
    
    def local_search_swap(current_set):
        """Try to improve current_set by swapping one element for another."""
        current_products = build_products(current_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i, elem in enumerate(current_set):
                # Try removing elem and adding a different candidate
                remaining = current_set[:i] + current_set[i+1:]
                remaining_products = build_products(remaining)
                for x in candidates:
                    if x in remaining:
                        continue
                    success, new_set, new_products = try_add_to_set(remaining, remaining_products, x)
                    if success:
                        current_set = new_set
                        current_products = new_products
                        improved = True
                        break
                if improved:
                    break
        return current_set
    
    random.seed(42)
    
    # Phase 1: Randomized greedy with multiple restarts
    for restart in range(500):
        if time.time() > deadline - 2:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = greedy_from_order(shuffled)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 15:
                break
    
    # Phase 2: Local search - try to add or swap elements
    if time.time() < deadline - 3:
        # Try to add elements to best_set
        current_set = local_search_add(best_set[:])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 3: More aggressive search with different heuristics
    if time.time() < deadline - 5:
        # Try sorted by different criteria
        sort_keys = [
            lambda x: x,  # increasing
            lambda x: -x,  # decreasing
            lambda x: len(prime_factors[x]),  # fewer prime factors first
            lambda x: -len(prime_factors[x]),  # more prime factors first
            lambda x: max(prime_factors[x].values(), default=0),  # highest exponent first
            lambda x: sum(prime_factors[x].values()),  # total prime factors
            lambda x: -sum(prime_factors[x].values()),  # fewer total prime factors
        ]
        for sort_key in sort_keys:
            if time.time() > deadline - 2:
                break
            
            sorted_candidates = sorted(candidates, key=sort_key)
            current_set = greedy_from_order(sorted_candidates)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
    
    # Phase 4: Swap-based local search on best set
    if time.time() < deadline - 4:
        current_set = local_search_swap(best_set[:])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 5: Final random restarts with time remaining, with local search polish
    while time.time() < deadline - 1:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = greedy_from_order(shuffled)
        
        # Apply local search add
        if time.time() < deadline - 1:
            current_set = local_search_add(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return [int(x) for x in best_set]
