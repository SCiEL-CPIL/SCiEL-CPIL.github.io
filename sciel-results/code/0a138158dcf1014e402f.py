import math

def construct():
    # Generate candidates 2..40
    candidates = list(range(2, 41))
    
    # Function to get prime factorization exponents
    def get_exponents(n):
        exps = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                exps[d] = exps.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            exps[n] = exps.get(n, 0) + 1
        return exps

    # Calculate norm (sum of exponents) for sorting
    # Larger norm means "heavier" prime factors, often better for independence
    # Actually, we want to maximize the count.
    # Greedy by largest norm first might help pack in the "dense" numbers (powers) first.
    
    scored_candidates = []
    for x in candidates:
        exps = get_exponents(x)
        norm = sum(exps.values())
        scored_candidates.append((norm, x))
        
    # Sort by norm descending, then by value descending (arbitrary tie-break)
    scored_candidates.sort(key=lambda item: (-item[0], -item[1]))
    
    products = {1}
    selected = []
    
    for norm, x in scored_candidates:
        new_products = set(products)
        for p in products:
            new_products.add(x * p)
        if len(new_products) == 2 * len(products):
            products = new_products
            selected.append(x)
            
    return selected
