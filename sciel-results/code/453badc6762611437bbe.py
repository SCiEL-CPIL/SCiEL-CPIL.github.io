"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Generate powerful numbers up to L
    # A powerful number has form prod(p_i^e_i) where e_i >= 2
    # Equivalently, n = a^2 * b where b is cube-free... no.
    # Better: n = prod(p_i^{2*k_i}) for k_i >= 1, OR n = prod(p_i^{e_i}) with e_i >= 2
    # Standard generation: for each "core" c that is a perfect square of a squarefree number... 
    # Actually, simplest: n is powerful iff for every prime p|n, p^2|n
    # Generation: n = m^2 * k where k is a product of distinct primes each appearing to odd power >= 1... no
    # Let me use: n = a^2 * b^3 where a,b are any positive integers? No that's not right either.
    # 
    # Correct characterization: n is powerful iff n = a^2 * b where b is cube-free and for each prime p|b, p^2|n.
    # Actually the cleanest: every powerful number can be written uniquely as a^2 * b^3 where a and b are coprime? No.
    #
    # Let me just use: n is powerful iff n = prod p_i^{e_i} with e_i >= 2.
    # To generate all powerful numbers <= L:
    # n = c^2 * d where d is a product of distinct primes (each to power 1 in d, so power 2 in n)... no.
    #
    # Simplest correct method: 
    # A powerful number n can be written as n = a^2 * b where b is cube-free and gcd(a,b)=1? No.
    #
    # Let me use the standard approach:
    # n is powerful iff n = prod_{p|n} p^{e_p} with e_p >= 2.
    # Generate by: for each "square part" s (perfect square), multiply by a "cube-free part" t where t is a product of distinct primes not in s, and t is squarefree... 
    #
    # Actually, the cleanest generation:
    # Every powerful number n can be uniquely written as n = a^2 * b^3 where a and b are coprime positive integers.
    # Wait, that's not standard. Let me verify: if n = prod p_i^{e_i}, e_i >= 2.
    # Write e_i = 2*a_i + 3*b_i? Not unique.
    #
    # Alternative: n = c^2 * d where d is a product of distinct primes, each appearing exactly once in d (so twice in n).
    # No, that only gives numbers where all exponents are even or... 
    #
    # Let me use a different approach. A powerful number has all prime factors with exponent >= 2.
    # So n = prod p_i^{e_i}, e_i >= 2.
    # We can write n = (prod p_i^{floor(e_i/2)})^2 * (prod p_i^{e_i mod 2})... but e_i mod 2 is 0 or 1, and if it's 1, exponent is odd >= 3.
    # 
    # Better: n = a^2 * b where b is a product of distinct primes, each to an odd power >= 1... this is getting complicated.
    #
    # Let me just use: generate all numbers of the form a^2 * b^3 <= L, where a >= 1, b >= 1, and additionally include 1.
    # Claim: every powerful number can be written as a^2 * b^3 with a,b positive integers.
    # Proof: n = prod p_i^{e_i}, e_i >= 2. For each p_i, write e_i = 2*x_i + 3*y_i where we want x_i,y_i >= 0.
    # Since e_i >= 2, we can always do this: if e_i is even, y_i=0, x_i=e_i/2. If e_i is odd >= 3, y_i=1, x_i=(e_i-3)/2 >= 0.
    # So yes, n = (prod p_i^{x_i})^2 * (prod p_i^{y_i})^3 = a^2 * b^3.
    # But this representation is not unique. That's fine, we just need to generate the set.
    
    L = 10**8  # Start with 10^8, can increase if time permits
    
    # Generate all a^2 * b^3 <= L
    powerful_set = set()
    powerful_set.add(1)
    
    # Max b: b^3 <= L => b <= L^(1/3)
    max_b = int(round(L ** (1/3))) + 1
    # For each b, max a: a^2 <= L / b^3 => a <= sqrt(L / b^3)
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    powerful = sorted(powerful_set)
    powerful_set = set(powerful)
    n_powerful = len(powerful)
    
    if time.time() > deadline - 60.0:
        # Not enough time for pair counting, try smaller L
        L = 10**7
        powerful_set = set([1])
        max_b = int(round(L ** (1/3))) + 1
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful_set.add(n)
        powerful = sorted(powerful_set)
        powerful_set = set(powerful)
        n_powerful = len(powerful)
    
    # Now find N with most representations as a+b where a,b powerful, a<=b
    # For each pair (a,b) with a<=b, a+b=N, count occurrences of N
    from collections import defaultdict
    count = defaultdict(int)
    
    # To save time, only consider pairs where a <= b and a+b <= 10^18
    # But a,b <= L, so a+b <= 2L <= 2*10^8 which is < 10^18, so all pairs are valid
    
    # Iterate over pairs
    n_p = len(powerful)
    for i in range(n_p):
        a = powerful[i]
        if time.time() > deadline - 30.0:
            break
        for j in range(i, n_p):
            b = powerful[j]
            N = a + b
            if N > 10**18:
                break
            count[N] += 1
    
    if not count:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find N with maximum count
    best_N = max(count, key=count.get)
    best_count = count[best_N]
    
    # Collect the actual pairs
    pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
