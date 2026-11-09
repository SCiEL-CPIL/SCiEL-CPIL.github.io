"""Rewritten program: maximum distinct-subset-product set in 2..40 via branch-and-bound."""

def construct():
    candidates = list(range(2, 41))
    n = len(candidates)
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def factorize(num):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while num % p == 0:
                vec[i] += 1
                num //= p
        return tuple(vec)
    
    vecs = [factorize(x) for x in candidates]
    
    def verify_set(indices):
        if not indices:
            return True
        nums = [candidates[i] for i in indices]
        k = len(nums)
        products = set()
        current = [1]
        for num in nums:
            new_products = [p * num for p in current]
            for np in new_products:
                if np in products:
                    return False
            products.update(new_products)
            current = new_products
        return True
    
    # Start with primes as initial solution
    prime_indices = [i for i in range(n) if sum(vecs[i]) == 1]
    composite_indices = [i for i in range(n) if i not in prime_indices]
    composite_indices.sort(key=lambda i: sum(vecs[i]))
    
    initial_set = list(prime_indices)
    for ci in composite_indices:
        test_set = initial_set + [ci]
        if verify_set(test_set):
            initial_set = test_set
    
    best_size = len(initial_set)
    best_set = list(initial_set)
    
    # Branch and bound with incremental product tracking
    order = prime_indices + composite_indices
    
    def dfs(pos, selected, products):
        nonlocal best_size, best_set
        if pos >= n:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = list(selected)
            return
        if len(selected) + (n - pos) <= best_size:
            return
        
        idx = order[pos]
        val = candidates[idx]
        
        # Try adding idx
        new_products = set(products)
        valid = True
        for p in products:
            np = p * val
            if np in new_products:
                valid = False
                break
            new_products.add(np)
        if valid:
            dfs(pos + 1, selected + [idx], frozenset(new_products))
        
        # Try skipping idx
        dfs(pos + 1, selected, products)
    
    dfs(0, [], frozenset({1}))
    
    result = best_set
    if not verify_set(result):
        result = initial_set
    
    return result
