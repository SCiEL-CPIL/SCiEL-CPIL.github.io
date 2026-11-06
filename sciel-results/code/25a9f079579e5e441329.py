import time
import math

def construct():
    start_time = time.time()
    deadline = 148.0  # Leave 2s buffer for evaluation overhead
    
    # Primes p = 1 mod 4
    # We need the smallest ones to maximize k for N <= 10^18
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421]
    
    # Helper to find a, b such that p = a^2 + b^2
    def sum_of_two_squares_prime(p):
        a = 1
        while a*a < p:
            b2 = p - a*a
            b = math.isqrt(b2)
            if b*b == b2:
                return a, b
            a += 1
        return None, None

    # Build N
    N = 1
    factors = [] # List of (a, b) for each prime
    for p in primes_1mod4:
        # Check if N * p <= 10^18
        if N * p > 10**18:
            break
        N *= p
        a, b = sum_of_two_squares_prime(p)
        if a is None:
            continue
        factors.append((a, b))
        
    if not factors:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    k = len(factors)
    
    # Generate all 2^k combinations
    # Each factor (a, b) corresponds to Gaussian integer z = a + bi or conjugate a - bi
    # We represent the choice as a sign for the imaginary part
    # If we pick z_i, the imaginary part sign is +1 (relative to a)
    # If we pick conj(z_i), the imaginary part sign is -1
    # Actually, we can just iterate through bitmasks.
    
    pairs_set = set()
    
    # Optimization: 2^k might be large if k is large.
    # k=11 -> 2048 iterations. Very fast.
    # k=20 -> 1 million iterations. Still fast in Python if simple.
    # But N limit restricts k to ~11-12.
    
    for mask in range(1 << k):
        x = 1
        y = 0
        for i in range(k):
            a, b = factors[i]
            # If bit i is 0, use a + bi
            # If bit i is 1, use a - bi
            # Note: Multiplying (x + yi) by (a + bi) or (a - bi)
            if (mask >> i) & 1:
                # Multiply by a - bi
                # (x + yi)(a - bi) = (xa + yb) + i(ya - xb)
                new_x = x * a + y * b
                new_y = y * a - x * b
            else:
                # Multiply by a + bi
                # (x + yi)(a + bi) = (xa - yb) + i(xb + ya)
                new_x = x * a - y * b
                new_y = x * b + y * a
            
            x = new_x
            y = new_y
        
        # Normalize x, y to be positive for the square values
        # The pair is {x^2, y^2}
        # We want a <= b, so we sort the squares
        sq_x = x * x
        sq_y = y * y
        
        if sq_x > sq_y:
            sq_x, sq_y = sq_y, sq_x
            
        # Ensure distinct and valid
        # a, b must be powerful. Squares are powerful.
        # a + b = N.
        # Check sum just in case of overflow or logic error (though math says it's N)
        if sq_x + sq_y == N:
            pairs_set.add((sq_x, sq_y))
            
    if not pairs_set:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Convert to list of lists
    pairs_list = [[a, b] for a, b in pairs_set]
    
    # Verify time
    if time.time() > deadline:
        # Fallback if somehow slow (shouldn't be)
        return {'N': N, 'pairs': pairs_list[:100]} # Truncate if needed, but shouldn't happen

    return {'N': N, 'pairs': pairs_list}
