import math

def construct():
    N = 1000000
    
    def get_phi(n):
        if n == 1:
            return 1
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    for n in range(1, N + 1):
        if get_phi(n) % 120 == 0:
            return n
            
    return -1
