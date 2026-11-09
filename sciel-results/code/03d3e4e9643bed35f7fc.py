def construct():
    import time
    start = time.time()
    deadline = start + 80

    def _max_prime_factor(n):
        max_pf = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                while n % d == 0:
                    n //= d
                max_pf = max(max_pf, d)
            d += 1
        if n > 1:
            max_pf = max(max_pf, n)
        return max_pf

    candidates = list(range(2, 41))
    
    # Heuristic: Sort by max prime factor descending, then value descending
    # This puts "safe" numbers (with large unique primes) first.
    candidates.sort(key=lambda x: (-_max_prime_factor(x), -x))

    best_set = []
    best_size = 0
    found_max = False

    def _search(current_set, product_set, remaining):
        nonlocal best_set, best_size, found_max
        
        if found_max:
            return
            
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
            if best_size == 18:
                found_max = True
                return

        if time.time() > deadline:
            return

        if not remaining:
            return

        for i, x in enumerate(remaining):
            if found_max or time.time() > deadline:
                break
            
            # Check validity
            valid = True
            new_products = []
            # We iterate over current products
            # If any p * x is in product_set, invalid
            for p in product_set:
                np = p * x
                if np in product_set:
                    valid = False
                    break
                new_products.append(np)
            
            if valid:
                # Construct new product set
                # product_set is a set of ints
                # new_product_set = product_set | set(new_products)
                # This is O(2^k).
                new_product_set = product_set.union(new_products)
                
                _search(current_set + [x], new_product_set, remaining[i+1:])

    _search([], {1}, candidates)

    if best_set:
        return [int(x) for x in best_set]
    return [2]
