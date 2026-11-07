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
    
    def check_valid_with_exponents(current_set, candidate, prime_factors):
        """Check if adding candidate to current_set keeps all subset products distinct.
        Uses exponent vectors for efficiency."""
        # Build exponent vector for candidate
        cand_vec = prime_factors[candidate]
        
        # For each existing subset product, check if multiplying by candidate creates a collision
        # We need: for all subsets A, B of current_set, if product(A) * candidate = product(B), then A must include candidate's "effect"
        # Simpler: check if any existing product equals candidate * some other product
        # This is equivalent to: candidate is not in the multiplicative span of current_set in a way that creates collisions
        
        # Efficient check: build all 2^k products and check
        products = {1}
        for x in current_set:
            new_prods = set()
            for p in products:
                np = p * x
                new_prods.add(np)
            products.update(new_prods)
        
        # Now check if adding candidate causes collision
        # New products would be {p * candidate for p in products}
        # Collision if p * candidate == q for some p, q in products with p != q/candidate
        # I.e., if candidate * p is in products for some p in products, and candidate * p != p (which is always true for candidate >= 2)
        # More precisely: if there exist p1, p2 in products such that p1 * candidate = p2, then collision
        # This happens iff candidate divides some product in products and the quotient is also in products
        
        for p in products:
            if p * candidate in products:
                return False
        return True
    
    random.seed(42)
    
    # Phase 1: Randomized greedy with multiple restarts
    for restart in range(500):
        if time.time() > deadline - 2:
            break
            
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_products = {1}
        
        for x in shuffled:
            # Check validity
            valid = True
            new_products = []
            for p in current_products:
                np = p * x
                if np in current_products:
                    valid = False
                    break
                new_products.append(np)
            
            if valid:
                current_set.append(x)
                current_products.update(new_products)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 15:
                break
    
    # Phase 2: Local search - try to add or swap elements
    if time.time() < deadline - 3:
        # Try to add elements to best_set
        current_set = best_set[:]
        current_products = {1}
        for x in current_set:
            new_prods = [p * x for p in current_products]
            current_products.update(new_prods)
        
        # Try adding each candidate not in current_set
        for x in candidates:
            if x in current_set:
                continue
            valid = True
            new_prods = []
            for p in current_products:
                np = p * x
                if np in current_products:
                    valid = False
                    break
                new_prods.append(np)
            
            if valid:
                current_set.append(x)
                current_products.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 3: More aggressive search with different heuristics
    if time.time() < deadline - 5:
        # Try sorted by different criteria
        for sort_key in [
            lambda x: x,  # increasing
            lambda x: -x,  # decreasing
            lambda x: len(prime_factors[x]),  # fewer prime factors first
            lambda x: -len(prime_factors[x]),  # more prime factors first
            lambda x: max(prime_factors[x].values(), default=0),  # highest exponent first
        ]:
            if time.time() > deadline - 2:
                break
            
            sorted_candidates = sorted(candidates, key=sort_key)
            current_set = []
            current_products = {1}
            
            for x in sorted_candidates:
                valid = True
                new_prods = []
                for p in current_products:
                    np = p * x
                    if np in current_products:
                        valid = False
                        break
                    new_prods.append(np)
                
                if valid:
                    current_set.append(x)
                    current_products.update(new_prods)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
    
    # Phase 4: Final random restarts with time remaining
    start_time = time.time()
    while time.time() - start_time < (deadline - time.time() - 1):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current_set = []
        current_products = {1}
        
        for x in shuffled:
            valid = True
            new_prods = []
            for p in current_products:
                np = p * x
                if np in current_products:
                    valid = False
                    break
                new_prods.append(np)
            
            if valid:
                current_set.append(x)
                current_products.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return [int(x) for x in best_set]
