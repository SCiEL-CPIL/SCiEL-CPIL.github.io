import math

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Strategy:
    1. Verify mathematically that 143 is the minimum.
       phi(n) must be a multiple of 120 = 2^3 * 3 * 5.
       For n < 143, we can check constraints or simply compute.
       phi(143) = phi(11*13) = 10*12 = 120.
       
    2. To be robust and avoid hardcoding a potentially incorrect value (though 143 is correct),
       we perform a fast search up to a small limit using a precomputed prime sieve.
       This ensures that if our mathematical assumption were wrong (it isn't), we would still find the answer quickly.
       
    3. The search is extremely fast (limit ~200), so we don't need complex deadline guards for the search itself,
       but we include a safety check just in case the environment is pathological.
    """
    
    # Upper bound for the search. We know 143 is the answer, so 200 is plenty.
    search_limit = 200
    
    # Sieve of Eratosthenes for primes up to sqrt(search_limit) is enough for phi calculation,
    # but we can just use trial division by small primes or a simple sieve up to search_limit.
    # For n <= 200, a simple list of primes is sufficient.
    
    def get_primes_up_to(limit):
        if limit < 2:
            return []
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, flag in enumerate(is_prime) if flag]
    
    primes = get_primes_up_to(search_limit)
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                while m % p == 0:
                    m //= p
                result -= result // p
        if m > 1:
            result -= result // m
        return result

    # Search for the minimal n
    for n in range(1, search_limit + 1):
        if phi(n) % 120 == 0:
            return n
            
    # Fallback: 143 is the mathematically proven minimum
    return 143
