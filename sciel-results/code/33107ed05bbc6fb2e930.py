    def phi(n):
        if n == 1:
            return 1
        result = n
        # Handle 2
        if n % 2 == 0:
            while n % 2 == 0:
                n //= 2
            result -= result // 2
        # Handle odd factors
        p = 3
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 2
        if n > 1:
            result -= result // n
        return result
