"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Generate primes up to 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(256)
    
    # Start with all primes plus 1
    # Greedily merge coprime pairs into their product if beneficial
    # We represent the set as a list of numbers
    selected = [1] + primes[:]
    
    # Repeatedly try to merge pairs
    merged = True
    while merged and (time.time() - start_time) < deadline:
        merged = False
        # Try all pairs
        i = 0
        while i < len(selected) and (time.time() - start_time) < deadline:
            j = i + 1
            while j < len(selected) and (time.time() - start_time) < deadline:
                a = selected[i]
                b = selected[j]
                prod = a * b
                if prod <= 256 and prod > a + b:
                    # Check that prod is coprime to all other elements
                    # Since a and b are coprime to all others (by invariant), 
                    # prod = a*b is also coprime to all others
                    # Remove a and b, add prod
                    selected.pop(j)
                    selected.pop(i)
                    selected.append(prod)
                    merged = True
                    # Restart scanning from beginning
                    i = 0
                    j = i + 1
                    break
                j += 1
            else:
                i += 1
                continue
            break
    
    # Ensure all are Python ints
    result = [int(x) for x in selected]
    return result

# EVOLVE-BLOCK-END
