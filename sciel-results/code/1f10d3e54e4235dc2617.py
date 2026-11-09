def construct():
    import sys
    from math import gcd
    import time
    
    N = 512
    
    # The set of divisors of a number M is a valid set if M <= N.
    # This is because for any a, b dividing M, lcm(a, b) also divides M.
    # Thus lcm(a, b) <= M <= N.
    # To maximize the size of the set, we should find M <= 512 with the maximum number of divisors.
    # For N=512, the number 504 has 24 divisors.
    # 504 = 2^3 * 3^2 * 7^1
    # Number of divisors = (3+1)*(2+1)*(1+1) = 4*3*2 = 24.
    # Other candidates like 480, 420, 360 also have 24 divisors.
    # We can just use 504.
    
    M = 504
    divisors = []
    for i in range(1, int(M**0.5) + 1):
        if M % i == 0:
            divisors.append(i)
            if i != M // i:
                divisors.append(M // i)
    
    # All divisors of 504 are <= 504 <= 512.
    # Return as a list
    return divisors
