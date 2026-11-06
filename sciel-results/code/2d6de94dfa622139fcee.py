def phi(x):
    if x <= 1:
        return 1 if x == 1 else 0
    result = x
    y = x
    for p in small_primes:
        if p * p > y:
            break
        if y % p == 0:
            while y % p == 0:
                y //= p
            result -= result // p
    if y > 1:
        # y is either prime or product of two primes > 10^6
        if is_prime_miller(y):
            result -= result // y
        else:
            # y = p * q where p, q > 10^6 and both prime
            d = pollard_rho(y)
            q = y // d
            result -= result // d
            result -= result // q
    return result
