"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin
    
    # Find all primes up to 256
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(n)
    
    # For each prime, include the largest multiple <= n
    selected = [1]
    for p in primes:
        largest_multiple = (n // p) * p
        selected.append(largest_multiple)
    
    # Verify pairwise coprimality (safety check)
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if math.gcd(selected[i], selected[j]) != 1:
                # Should not happen, but fallback to greedy if it does
                break
        else:
            continue
        break
    else:
        # All pairs are coprime
        return [int(x) for x in selected]
    
    # Fallback: greedy approach (should never be reached)
    selected_fb = [1]
    for a in range(n, 1, -1):
        if time.time() > deadline:
            break
        if all(math.gcd(a, b) == 1 for b in selected_fb):
            selected_fb.append(a)
    return [int(x) for x in selected_fb]
# EVOLVE-BLOCK-END
