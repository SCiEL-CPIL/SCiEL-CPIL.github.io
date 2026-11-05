def is_valid(seq):
    n = len(seq)
    prods = set()
    for i in range(n):
        prod_vec = [0]*num_primes
        for j in range(i, n):
            # add exp of seq[j]
            # ...
            if tuple(prod_vec) in prods:
                return False
            prods.add(tuple(prod_vec))
    return True
