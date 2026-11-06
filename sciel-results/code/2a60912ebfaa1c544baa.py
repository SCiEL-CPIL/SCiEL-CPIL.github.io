import math

def construct():
    target = 120
    for n in range(1, 1000001):
        # Calculate phi(n)
        phi = n
        x = n
        p = 2
        while p * p <= x:
            if x % p == 0:
                while x % p == 0:
                    x //= p
                phi -= phi // p
            p += 1
        if x > 1:
            phi -= phi // x
        
        if phi % target == 0:
            return n
    return 1000000 # Fallback
