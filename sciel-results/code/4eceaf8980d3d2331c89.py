"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 40, 'maximum_size': 18}

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Generate primes from 2 to 40
    primes = [i for i in range(2, P['n'] + 1) if _is_prime(i)]
    
    # Find prime powers <= 40
    prime_powers = []
    for p in primes:
        power = p
        while power <= P['n']:
            prime_powers.append(power)
            power *= p
    
    # The prime powers form a basis with distinct subset products
    # Sort them
    prime_powers.sort()
    
    # All primes (except 2) are prime powers (p^1)
    # 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32
    # 3^1=3, 3^2=9, 3^3=27
    # 5^1=5, 5^2=25
    # 7^1=7
    # 11, 13, 17, 19, 23, 29, 31, 37
    
    # So prime_powers = [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37]
    # That's 19 elements! But we're limited to 18.
    
    # Let's verify: each prime power p^k has a unique prime factorization, so subset products are distinct.
    # This is the key insight: the set of all prime powers <= 40 has distinct subset products.
    
    # Count them:
    # 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32  (5)
    # 3^1=3, 3^2=9, 3^3=27                   (3)
    # 5^1=5, 5^2=25                          (2)
    # 7^1=7                                  (1)
    # 11, 13, 17, 19, 23, 29, 31, 37         (8)
    # Total: 5+3+2+1+8 = 19
    
    # We can select up to 18. Let's take 18 of them.
    # Which one to exclude? To maximize the set, we should exclude the largest or keep it simple.
    # Actually, we just need 18 distinct integers. Let's take the first 18.
    
    selected = prime_powers[:18]
    
    # Verify all subset products are distinct (for safety)
    # For small sets, we can verify
    if len(selected) <= 20:
        products = [1]
        for x in selected:
            new_products = [p * x for p in products]
            products.extend(new_products)
        if len(products) != len(set(products)):
            # Fallback: just return primes
            return [p for p in primes[:18]]
    
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
