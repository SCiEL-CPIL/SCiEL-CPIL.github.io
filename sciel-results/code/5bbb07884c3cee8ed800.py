import time
import random
from math import gcd
from collections import Counter

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
    
    # Direct construction using primes
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    def can_add(current_products, candidate):
        """Check if adding candidate keeps all subset products distinct."""
        for p in current_products:
            if p * candidate in current_products:
                return False
        return True
    
    def build_greedy(order):
        """Build a set greedily following the given order."""
        current_set = []
        current_products = {1}
        
        for x in order:
            if can_add(current_products, x):
                current_set.append(x)
                current_products.update(p * x for p in current_products)
        
        return current_set, current_products
    
    def local_search_swap(current_set, current_products, candidates, time_limit):
        """Try to improve by swapping elements or adding new ones."""
        best_local = current_set[:]
        best_local_products = set(current_products)
        
        # Try adding elements not in current_set
        for x in candidates:
            if x not in current_set:
                if can_add(best_local_products, x):
                    best_local.append(x)
                    best_local_products.update(p * x for p in best_local_products)
        
        # Try swaps: remove one, add two
        for i in range(len(best_local)):
            if time.time() > time_limit:
                break
            removed = best_local[i]
            # Rebuild products without removed
            temp_set = best_local[:i] + best_local[i+1:]
            temp_products = {1}
            for x in temp_set:
                temp_products.update(p * x for p in temp_products)
            
            # Try adding pairs
            for x in candidates:
                if x in temp_set:
                    continue
                if can_add(temp_products, x):
                    new_set = temp_set + [x]
                    new_products = set(temp_products)
                    new_products.update(p * x for p in temp_products)
                    
                    # Try adding one more
                    for y in candidates:
                        if y in new_set or y == removed:
                            continue
                        if can_add(new_products, y):
                            final_set = new_set + [y]
                            if len(final_set) > len(best_local):
                                best_local = final_set[:]
                                final_products = set(new_products)
                                final_products.update(p * y for p in new_products)
                                best_local_products = final_products
                                break
        
        return best_local, best_local_products
    
    random.seed(42)
    
    # Phase 1: Heuristic orderings
    heuristic_orders = [
        sorted(candidates, key=lambda x: (len(prime_factors[x]), sum(prime_factors[x].values()), x)),
        sorted(candidates, key=lambda x: (-x,)),  # decreasing
        sorted(candidates, key=lambda x: (x,)),  # increasing
        sorted(candidates, key=lambda x: (max(prime_factors[x].values(), default=0), x)),
        sorted(candidates, key=lambda x: (min(prime_factors[x].values(), default=1), x)),
    ]
    
    for order in heuristic_orders:
        if time.time() > deadline - 3:
            break
        s, _ = build_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
    
    # Phase 2: Randomized greedy with multiple restarts
    for restart in range(2000):
        if time.time() > deadline - 3:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        s, _ = build_greedy(shuffled)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            if best_size >= 16:
                break
    
    # Phase 3: Local search on best set
    if time.time() < deadline - 3:
        _, best_products = build_greedy(best_set)
        improved_set, improved_products = local_search_swap(best_set, best_products, candidates, deadline - 1)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    # Phase 4: More random restarts with time remaining
    start_time = time.time()
    remaining = deadline - time.time() - 1
    if remaining > 0:
        while time.time() - start_time < remaining:
            shuffled = candidates[:]
            random.shuffle(shuffled)
            
            s, _ = build_greedy(shuffled)
            
            if len(s) > best_size:
                best_size = len(s)
                best_set = s[:]
    
    return [int(x) for x in best_set]
