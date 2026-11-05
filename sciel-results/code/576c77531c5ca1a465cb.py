    def phi(x):
        if x in phi_cache:
            return phi_cache[x]
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        phi_cache[x] = result
        return result
