import math

def phi(n):
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

def construct():
    target = 120
    # The minimal n is expected to be small.
    # We iterate upwards.
    for n in range(1, 1000001):
        if phi(n) % target == 0:
            return n
    return 1000000
