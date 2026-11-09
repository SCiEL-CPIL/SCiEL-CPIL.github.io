    def is_smooth(n, primes):
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1
