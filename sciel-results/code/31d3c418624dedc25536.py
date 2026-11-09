"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """Return a list of distinct ints from 1..256, pairwise coprime, maximizing sum."""
    deadline = time.time() + 50  # safety margin before 60s kill
    N = 256
    
    # Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # For each prime, find the largest power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    result = [1] + list(prime_powers.values())
    
    # Systematic local search: try replacing pairs of elements with a single
    # larger coprime integer, and adding single elements.
    # Precompute prime factorization for 1..256
    def factorize(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    factors_map = {i: factorize(i) for i in range(1, N + 1)}
    
    def is_coprime_with_set(x, s):
        fx = factors_map[x]
        for y in s:
            if fx & factors_map[y]:
                return False
        return True

    improved = True
    while improved:
        improved = False
        current_sum = sum(result)
        result_set = set(result)
        
        # Try adding a single element
        best_add = None
        best_add_gain = 0
        for x in range(N, 0, -1):
            if x not in result_set and is_coprime_with_set(x, result_set):
                gain = x
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add = x
        if best_add is not None:
            result.append(best_add)
            result_set.add(best_add)
            improved = True
            continue
        
        # Try replacing two elements with one larger element
        best_swap = None
        best_swap_gain = 0
        n = len(result)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = result[i], result[j]
                removed_sum = a + b
                # Candidate x must be coprime with all elements except a and b
                remaining = [x for x in result if x != a and x != b]
                for x in range(N, 0, -1):
                    if x in result_set:
                        continue
                    # x must not share factors with a or b (they're being removed, but
                    # we need x coprime with remaining elements)
                    if is_coprime_with_set(x, set(remaining)):
                        gain = x - removed_sum
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (a, b, x)
                            break  # largest x for this pair
        if best_swap is not None:
            a, b, x = best_swap
            result.remove(a)
            result.remove(b)
            result.append(x)
            improved = True
    
    # Verify pairwise coprime (sanity check)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback: return just prime powers for safety
                return [1] + list(prime_powers.values())
    
    return result
# EVOLVE-BLOCK-END
