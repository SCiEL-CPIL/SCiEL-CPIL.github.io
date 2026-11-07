"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # Structural approach: Use exponent-vector representation to verify multiplicative independence.
    # Each number in [2,40] maps to a vector of exponents for primes in [2,40].
    # Distinct subset products <=> the exponent vectors are linearly independent over Q.
    # The rank of the exponent matrix gives the maximum independent set size.
    
    primes_in_range = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Build exponent vectors for each number in [2, 40]
    def get_exponent_vector(n, primes):
        vec = []
        for p in primes:
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            vec.append(count)
        return tuple(vec)
    
    # The primes themselves form a basis: each has a 1 in its own position and 0 elsewhere.
    # This gives us 12 linearly independent vectors, which is the maximum possible rank
    # since we only have 12 primes in the range.
    
    solution = list(primes_in_range)
    
    # Verify using direct product enumeration (fast for k=12: 4096 products)
    k = len(solution)
    products = set()
    valid = True
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= solution[i]
        if prod in products:
            valid = False
            break
        products.add(prod)
    
    if not valid:
        # Should never happen for primes, but fallback gracefully
        return [2, 3, 5, 7, 11, 13]
    
    return solution

# EVOLVE-BLOCK-END
