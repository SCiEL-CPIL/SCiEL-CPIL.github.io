"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 40, 'maximum_size': 18}

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Candidate numbers are 2 to 40
    candidates = list(range(2, 41))
    
    # We will use backtracking to find the largest subset with distinct subset products.
    # State: index in candidates, current set of subset products (as a set of ints), current selected list.
    
    best_solution = []
    
    # To speed up, we can order candidates. 
    # Numbers with fewer prime factors are "lighter" and less likely to cause collisions?
    # Or maybe just sort by value.
    # Let's sort by number of prime factors (with multiplicity) ascending, then value.
    def count_prime_factors(n):
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
        
    candidates.sort(key=lambda x: (count_prime_factors(x), x))
    
    def backtrack(idx, current_products, current_selected):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        # Update best if we found a larger set
        if len(current_selected) > len(best_solution):
            best_solution = list(current_selected)
            
        # If we have exhausted candidates, return
        if idx == len(candidates):
            return
            
        # Option 1: Skip candidates[idx]
        backtrack(idx + 1, current_products, current_selected)
        
        # Option 2: Include candidates[idx] if no collision
        num = candidates[idx]
        # Check if adding num causes a collision
        # New products will be {p * num for p in current_products}
        # Collision if any p * num is already in current_products
        # Or if any p1 * num == p2 * num' ? No, we just add new products to the set.
        # The set of products must remain distinct.
        # Current products: S. New products: S' = S union {p*num for p in S}.
        # We need S' to have distinct elements.
        # This is equivalent to: 
        # 1. No p in S such that p*num is in S.
        # 2. No p1, p2 in S such that p1*num == p2*num (impossible if num!=0)
        # 3. No p1, p2 in S such that p1 == p2*num (covered by 1 if we check all)
        # Actually, simply checking if any new product exists in the old set is sufficient?
        # Let S be the set of subset products of the current selection.
        # If we add x, the new set of products is S union {s*x for s in S}.
        # For these to be distinct, we must have:
        # a) No s in S such that s*x is in S. (Because s*x would be a duplicate of an existing product)
        # b) The new products {s*x for s in S} must be distinct among themselves. Since x is fixed and non-zero, s1*x == s2*x implies s1==s2, which is true since S has distinct elements.
        # So the condition is simply: for all s in S, s*x not in S.
        
        collision = False
        # Optimization: only check s such that s*x <= max(S) is not enough, s*x can be larger.
        # We just iterate through S.
        for s in current_products:
            if (s * num) in current_products:
                collision = True
                break
                
        if not collision:
            new_products = set(current_products)
            for s in current_products:
                new_products.add(s * num)
            current_selected.append(num)
            backtrack(idx + 1, new_products, current_selected)
            current_selected.pop()
            
    # Initial state: empty set of numbers, product set is {1}
    backtrack(0, {1}, [])
    
    if time.time() > deadline:
         # Fallback to a known good set or primes if search didn't finish
         # Primes are always valid
         primes = [i for i in range(2, 41) if _is_prime(i)]
         return [int(p) for p in primes]
         
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
