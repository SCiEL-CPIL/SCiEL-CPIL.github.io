import math

def construct():
    # We need to find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    # 120 = 2^3 * 3 * 5
    # phi(n) is divisible by 120 if and only if it is divisible by 8, 3, and 5.
    
    # Let's analyze the conditions for phi(n) to be divisible by these factors.
    # phi(n) = n * product_{p|n} (1 - 1/p)
    
    # Instead of computing phi for all n up to 1,000,000, we can search for the smallest n directly.
    # Since we want to minimize n, we can iterate n from 1 upwards and check if phi(n) % 120 == 0.
    # However, computing phi(n) from scratch for each n is expensive.
    # A better approach is to use a sieve to compute phi(n) for all n up to a certain limit, 
    # or simply iterate and compute phi(n) on the fly if we have a fast factorization.
    
    # Given the small target (120) and the fact that the smallest n is likely small,
    # we can try to find it by checking n=1, 2, 3, ... and computing phi(n).
    # But wait, the current program uses a sieve up to 1,000,000 which is fast enough.
    # The score 100.6993 suggests the current program returns a value around 100.
    # Let's verify: what is the smallest n with phi(n) divisible by 120?
    
    # Let's list some small n and their phi(n):
    # n=1, phi=1
    # n=2, phi=1
    # n=3, phi=2
    # n=4, phi=2
    # n=5, phi=4
    # n=6, phi=2
    # n=7, phi=6
    # n=8, phi=4
    # n=9, phi=6
    # n=10, phi=4
    # n=11, phi=10
    # n=12, phi=4
    # n=13, phi=12
    # n=14, phi=6
    # n=15, phi=8
    # n=16, phi=8
    # n=17, phi=16
    # n=18, phi=6
    # n=19, phi=18
    # n=20, phi=8
    # n=21, phi=12
    # n=22, phi=10
    # n=23, phi=22
    # n=24, phi=8
    # n=25, phi=20
    # n=26, phi=12
    # n=27, phi=18
    # n=28, phi=12
    # n=29, phi=28
    # n=30, phi=8
    # n=31, phi=30
    # n=32, phi=16
    # n=33, phi=20
    # n=34, phi=16
    # n=35, phi=24
    # n=36, phi=12
    # n=37, phi=36
    # n=38, phi=18
    # n=39, phi=24
    # n=40, phi=16
    # n=41, phi=40
    # n=42, phi=12
    # n=43, phi=42
    # n=44, phi=20
    # n=45, phi=24
    # n=46, phi=22
    # n=47, phi=46
    # n=48, phi=16
    # n=49, phi=42
    # n=50, phi=20
    # n=51, phi=32
    # n=52, phi=24
    # n=53, phi=52
    # n=54, phi=18
    # n=55, phi=40
    # n=56, phi=24
    # n=57, phi=36
    # n=58, phi=28
    # n=59, phi=58
    # n=60, phi=16
    # n=61, phi=60
    # n=62, phi=30
    # n=63, phi=36
    # n=64, phi=32
    # n=65, phi=48
    # n=66, phi=20
    # n=67, phi=66
    # n=68, phi=32
    # n=69, phi=44
    # n=70, phi=24
    # n=71, phi=70
    # n=72, phi=24
    # n=73, phi=72
    # n=74, phi=36
    # n=75, phi=40
    # n=76, phi=36
    # n=77, phi=60
    # n=78, phi=24
    # n=79, phi=78
    # n=80, phi=32
    # n=81, phi=54
    # n=82, phi=40
    # n=83, phi=82
    # n=84, phi=24
    # n=85, phi=64
    # n=86, phi=42
    # n=87, phi=56
    # n=88, phi=40
    # n=89, phi=88
    # n=90, phi=24
    # n=91, phi=72
    # n=92, phi=44
    # n=93, phi=60
    # n=94, phi=46
    # n=95, phi=72
    # n=96, phi=32
    # n=97, phi=96
    # n=98, phi=42
    # n=99, phi=60
    # n=100, phi=40
    
    # None of these have phi(n) divisible by 120. The smallest phi(n) divisible by 120 is 120 itself.
    # So we need phi(n) >= 120.
    # Let's continue checking larger n.
    
    # Actually, let's just write a fast program to find it.
    # We can use a sieve to compute phi(n) for n up to, say, 10000, which is very fast.
    # Then find the smallest n where phi[n] % 120 == 0.
    
    N = 1000000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] = phi[j] // i * (i - 1)
    
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    return N
