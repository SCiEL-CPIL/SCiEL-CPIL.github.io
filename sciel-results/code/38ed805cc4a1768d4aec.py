import itertools

def construct():
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Non-square-free numbers in 2..40
    # A number is square-free if no prime factor divides it more than once
    def is_square_free(n):
        d = 2
        while d * d <= n:
            if n % (d * d) == 0:
                return False
            d += 1
        return True

    candidates = [i for i in range(2, 41) if not is_square_free(i)]
    
    # Find largest subset of candidates with distinct subset products
    best_subset = []
    max_size = 0
    
    # Iterate all subsets of candidates
    # There are 2^14 = 16384 subsets, very fast
    for r in range(len(candidates), -1, -1):
        found = False
        for comb in itertools.combinations(candidates, r):
            # Check if subset products are distinct
            prods = set()
            valid = True
            # Generate all subset products of comb
            # Use bitmasks or itertools
            # Since r <= 14, 2^r <= 16384, total operations ~ 16384 * 16384 / 2 ~ 1.3e8 which might be slow in Python
            # But we can prune.
            
            # Actually, we can just check distinctness using a set of products.
            # Number of products is 2^r.
            # We can generate them recursively or iteratively.
            
            products = [1]
            for x in comb:
                new_prods = [p * x for p in products]
                products.extend(new_prods)
            
            if len(products) == len(set(products)):
                best_subset = list(comb)
                max_size = r
                found = True
                break
        if found:
            break
            
    return primes + best_subset
