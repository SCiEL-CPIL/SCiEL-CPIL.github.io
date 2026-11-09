def phi(n, primes):
    result = n
    y = n
    for p in primes:
        if p * p > y:
            break
        if y % p == 0:
            while y % p == 0:
                y //= p
            result -= result // p
    if y > 1:
        result -= result // y
    return result
