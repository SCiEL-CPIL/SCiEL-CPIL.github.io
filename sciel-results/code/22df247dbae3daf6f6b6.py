    def phi_large(n):
        result = n
        temp = n
        # Manual check for small primes to reduce loop iterations
        if temp % 2 == 0:
            result -= result // 2
            while temp % 2 == 0: temp //= 2
        if temp % 3 == 0:
            result -= result // 3
            while temp % 3 == 0: temp //= 3
        if temp % 5 == 0:
            result -= result // 5
            while temp % 5 == 0: temp //= 5
        if temp % 7 == 0:
            result -= result // 7
            while temp % 7 == 0: temp //= 7
        if temp % 11 == 0:
            result -= result // 11
            while temp % 11 == 0: temp //= 11
        if temp % 13 == 0:
            result -= result // 13
            while temp % 13 == 0: temp //= 13
        if temp % 17 == 0:
            result -= result // 17
            while temp % 17 == 0: temp //= 17
        if temp % 19 == 0:
            result -= result // 19
            while temp % 19 == 0: temp //= 19
        if temp % 23 == 0:
            result -= result // 23
            while temp % 23 == 0: temp //= 23
        if temp % 29 == 0:
            result -= result // 29
            while temp % 29 == 0: temp //= 29
        if temp % 31 == 0:
            result -= result // 31
            while temp % 31 == 0: temp //= 31
            
        # Continue with primes list starting from 37
        # We need to skip primes already checked
        # primes list: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37...
        # Index 11 is 37
        for p in primes[11:]:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result
