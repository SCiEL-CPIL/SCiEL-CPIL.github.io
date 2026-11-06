import sys
import time

def construct():
    start_time = time.time()
    deadline = 60.0
    
    N = 1000000
    
    # 1. Compute Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute Powerful Part for all numbers
    # powerful_part[x] will store the powerful part of x
    powerful_part = [1] * (N + 1)
    
    # We can compute this efficiently.
    # A number x has powerful part > 1 if it is divisible by p^2 for some p.
    # Alternatively, we can just factorize each number using spf.
    # Given N=1e6, factorization for each number is fast enough.
    
    for i in range(2, N + 1):
        x = i
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            power = 1
            while x % p == 0:
                x //= p
                e += 1
                power *= p
            if e >= 2:
                pp *= power
        powerful_part[i] = pp
        
    # 3. Find n in [10000, 1000000] maximizing PP(n) * PP(n+1)
    # Note: n+1 can be up to 1000001.
    # We need powerful_part up to 1000001.
    # Let's extend N to 1000001 to be safe.
    
    N_ext = 1000001
    spf = list(range(N_ext + 1))
    for i in range(2, int(N_ext**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N_ext + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    powerful_part = [1] * (N_ext + 1)
    for i in range(2, N_ext + 1):
        x = i
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            power = 1
            while x % p == 0:
                x //= p
                e += 1
                power *= p
            if e >= 2:
                pp *= power
        powerful_part[i] = pp

    best_n = 10000
    best_score = 0
    
    for n in range(10000, 1000000):
        score = powerful_part[n] * powerful_part[n+1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline - 5.0:
            break
            
    return best_n

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    N = 1000001
    
    # 1. Compute Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    # 2. Compute Powerful Part for all numbers
    powerful_part = [1] * (N + 1)
    
    for i in range(2, N + 1):
        x = i
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            power = 1
            while x % p == 0:
                x //= p
                e += 1
                power *= p
            if e >= 2:
                pp *= power
        powerful_part[i] = pp
        
    # 3. Find n in [10000, 1000000] maximizing PP(n) * PP(n+1)
    best_n = 10000
    best_score = 0
    
    for n in range(10000, 1000000):
        score = powerful_part[n] * powerful_part[n+1]
        if score > best_score:
            best_score = score
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n
# EVOLVE-BLOCK-END
