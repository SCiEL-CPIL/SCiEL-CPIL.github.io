    def is_p_smooth(n, p):
        if n == 1:
            return True
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for q in small_primes:
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
        if is_prime(n):
            return n <= p
            
        # n is composite
        # Find a factor
        d = pollard_rho(n)
        # Check d and n//d
        # Note: d might be composite
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)
