import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    n = len(candidates)
    
    # Precompute conflict graph: can[i][j] = True if i and j can coexist
    # Two numbers conflict if they share a prime factor OR form a multiplicative dependency
    # Actually, for subset products to be distinct, we need no two different subsets to have the same product.
    # A simpler sufficient condition: all numbers are pairwise coprime AND no product of a subset equals another number.
    # But the real condition is: the set must be "multiplicatively independent" in a strong sense.
    # 
    # Key insight: if we use exponent vectors (prime factorization), two subsets collide iff
    # their exponent vectors are equal. So the condition is that all 2^k exponent vectors are distinct.
    # This is equivalent to: no non-trivial multiplicative relation exists among the selected numbers.
    #
    # For the greedy approach, we need: can we add x to current_set without creating a collision?
    # This means: for every subset S of current_set, x * prod(S) must not equal prod(T) for any subset T.
    # Equivalently: x must not be in the ratio set {prod(S)/prod(T) : S,T subsets of current_set}
    # (when the division is exact).
    
    # Precompute factorizations
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
    
    factorizations = {x: factorize(x) for x in candidates}
    
    # Precompute which primes are relevant
    all_primes = set()
    for x in candidates:
        all_primes.update(factorizations[x].keys())
    all_primes = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    # Convert factorizations to sparse vectors (list of (prime_idx, exponent))
    vecs = {}
    for x in candidates:
        vecs[x] = [(prime_to_idx[p], e) for p, e in factorizations[x].items()]
    
    # Direct construction with primes
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    # Fast validity check: given current_set and candidate x, check if adding x keeps products distinct
    # Use the fact that we only need to check if x * prod(S) collides with prod(T) for any S, T subsets
    # This is equivalent to: x is not in the set of ratios prod(S)/prod(T) where S != T
    # For efficiency, track all subset products and check if any product * x equals an existing product
    
    def check_valid(current_set, x):
        """Check if x can be added to current_set without creating subset product collisions."""
        if not current_set:
            return True
        
        # Compute all subset products of current_set
        products = [1]
        for elem in current_set:
            new_prods = [p * elem for p in products]
            products.extend(new_prods)
        
        # Check if x * p equals any product for any p in products
        # i.e., x is in {prod / p : prod in products, p in products, prod % p == 0, prod/p == x}
        # Equivalently: for any p in products, if p * x is in products, collision!
        product_set = set(products)
        for p in products:
            if p * x in product_set:
                return False
        return True
    
    # Even faster: precompute for each candidate x, which other candidates conflict with it
    # Two candidates a, b conflict if there exist subsets S of {a} and T of {b} (trivially S={a} or {}, T={b} or {})
    # such that prod(S) = prod(T). For single elements: a=b (impossible since distinct) or a=1 (not in range).
    # So single pairs never conflict directly! The conflict arises from multiplicative dependencies
    # involving 3+ elements.
    
    # Better approach: use the ratio set. For a set S, define R(S) = {prod(A)/prod(B) : A,B subsets of S, prod(B) != 0}
    # A new element x can be added iff x not in R(S) and 1/x not in R(S) (i.e., x not in R(S) and x^{-1} not in R(S))
    # Actually: collision occurs iff x * prod(A) = prod(B) for some A,B subsets, i.e., x = prod(B)/prod(A)
    # So x can be added iff x is not in the ratio set R(S) = {prod(B)/prod(A) : A,B subsets of S}
    
    # Let's use a cleaner approach with the ratio set
    def compute_ratio_set(current_set):
        """Compute the set of all ratios prod(A)/prod(B) for subsets A,B of current_set."""
        products = [1]
        for elem in current_set:
            new_prods = [p * elem for p in products]
            products.extend(new_prods)
        
        ratios = set()
        for a in products:
            for b in products:
                if b != 0 and a % b == 0:
                    ratios.add(a // b)
        return ratios
    
    # For greedy, we'll use the direct product check which is O(2^k) per addition
    # but with k <= ~15 this is manageable
    
    random.seed(42)
    
    # Greedy with multiple orderings
    def greedy_search(ordering):
        current_set = []
        current_products = {1}
        
        for x in ordering:
            # Check validity
            product_set = set(current_products)
            valid = True
            for p in current_products:
                if p * x in product_set:
                    valid = False
                    break
            
            if valid:
                current_set.append(x)
                new_prods = {p * x for p in current_products}
                current_products.update(new_prods)
        
        return current_set
    
    # Try different orderings
    orderings = [
        sorted(candidates, key=lambda x: len(factorizations[x])),  # fewer prime factors first
        sorted(candidates),  # ascending
        sorted(candidates, reverse=True),  # descending
        candidates[:],  # natural order
    ]
    
    for ordering in orderings:
        result = greedy_search(ordering)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Randomized restarts
    for restart in range(3000):
        if time.time() > deadline - 2:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_search(shuffled)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Local search: try to improve by swaps
    if time.time() < deadline - 3:
        local_best = best_set[:]
        local_size = len(local_best)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Try adding each candidate
            for x in candidates:
                if x in local_best:
                    continue
                
                current_products = {1}
                for elem in local_best:
                    current_products.update({p * elem for p in current_products})
                
                product_set = set(current_products)
                valid = True
                for p in current_products:
                    if p * x in product_set:
                        valid = False
                        break
                
                if valid:
                    local_best.append(x)
                    local_size += 1
                    best_set = local_best[:]
                    best_size = local_size
                    improved = True
                    break
            
            # Try swaps
            if not improved:
                for i, old_x in enumerate(local_best):
                    trial_set = [x for j, x in enumerate(local_best) if j != i]
                    
                    for new_x in candidates:
                        if new_x in trial_set:
                            continue
                        
                        current_products = {1}
                        for elem in trial_set:
                            current_products.update({p * elem for p in current_products})
                        
                        product_set = set(current_products)
                        valid = True
                        for p in current_products:
                            if p * new_x in product_set:
                                valid = False
                                break
                        
                        if valid:
                            local_best = trial_set + [new_x]
                            improved = True
                            break
                    if improved:
                        break
    
    # More aggressive randomized search with time remaining
    start_time = time.time()
    max_time = max(0, deadline - time.time() - 1.0)
    while time.time() - start_time < max_time:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_search(shuffled)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Final verification and type check
    return [int(x) for x in best_set]
