import math

# Precompute primes for factorization
def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# We need to factorize B. B = n*x.
# n <= 256.
# If we search x up to, say, 10000, B <= 2,560,000.
# sqrt(2,560,000) ~ 1600.
# Primes up to 2000 are sufficient.

PRIMES = get_primes(2000)

def factorize(n):
    factors = {}
    temp = n
    for p in PRIMES:
        if p * p > temp:
            break
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors

def get_divisors_from_factors(factors):
    # factors: dict {prime: exponent}
    # We want divisors of n^2, so exponents are 2*exp
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        for d in divs:
            for k in range(2 * e + 1):
                new_divs.append(d * (p ** k))
        divs = new_divs
    return divs

def construct():
    answers = []
    # Search range for x. 
    # Current code searches 100 values. 
    # Let's search more, e.g., up to n + 1000 or fixed 10000?
    # But we must be fast.
    
    for n in range(3, 257):
        found = None
        start_x = n // 4 + 1
        
        # We will iterate x. 
        # To ensure we don't timeout, we limit x range.
        # But we want to maximize solved fraction.
        # Let's try x up to start_x + 5000?
        # With divisor approach, inner loop is fast.
        
        for x in range(start_x, start_x + 5000):
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            # Factorize B
            fac = factorize(B)
            
            # Get divisors of B^2
            # This can be many. 
            # Instead of generating ALL divisors, 
            # we can iterate and check congruence.
            
            divs = get_divisors_from_factors(fac)
            
            # Sort divisors? 
            # We want y, z distinct.
            # y = (U+B)/A, z = (V+B)/A.
            # If U=V, then y=z. We must avoid U=V.
            
            # Iterate through divisors
            for U in divs:
                if U > B2: # Should not happen
                    break
                V = B2 // U
                if U * V != B2:
                    continue
                
                # Check congruence: U = -B mod A
                if (U + B) % A != 0:
                    continue
                if (V + B) % A != 0:
                    continue
                
                y = (U + B) // A
                z = (V + B) // A
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check bounds
                if y > 10**12 or z > 10**12:
                    continue
                if y <= 0 or z <= 0:
                    continue
                
                # Verify
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
