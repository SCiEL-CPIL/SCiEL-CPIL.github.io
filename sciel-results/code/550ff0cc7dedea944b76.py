def sieve(n):
    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, n + 1, i):
                is_prime_arr[j] = False
    return [i for i, flag in enumerate(is_prime_arr) if flag]

SMALL_PRIMES = sieve(10000)

def max_prime_factor_fast(n, p_limit):
    # p_limit is p
    if n == 1:
        return 1
    max_f = 1
    temp_n = n
    for q in SMALL_PRIMES:
        if q > p_limit:
            break
        if q * q > temp_n:
            break
        while temp_n % q == 0:
            max_f = max(max_f, q)
            temp_n //= q
    if temp_n > 1:
        # If temp_n > 1, it has a prime factor > 10000 (or > p_limit if p_limit < 10000)
        # If p_limit < 10000, and we broke because q > p_limit, then temp_n has factor > p_limit
        if p_limit < 10000:
             # We need to be careful. If we broke because q > p_limit, 
             # and temp_n > 1, then temp_n has a prime factor > p_limit?
             # Yes, because we divided out all primes <= p_limit.
             return temp_n # Which is > p_limit
             
        if is_prime(temp_n):
            max_f = max(max_f, temp_n)
        else:
            factors = factorize(temp_n)
            max_f = max(max_f, max(factors))
    return max_f
