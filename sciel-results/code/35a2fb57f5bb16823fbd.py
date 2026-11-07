"""Combined program: prime-power baseline with greedy merge of compatible prime pairs."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # For each prime, find the largest prime power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Identify potential merges:
    # A merge replaces (P_p, P_q) with product p*q when p*q <= N and p*q > P_p + P_q
    # This is valid because p*q is coprime to all other selected numbers
    # (it only uses prime factors p and q, which are removed from their individual slots)
    merges = []
    for i in range(len(primes)):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, len(primes)):
            q = primes[j]
            Pq = prime_powers[q]
            prod = p * q
            if prod > N:
                break
            gain = prod - Pp - Pq
            if gain > 0:
                merges.append((gain, p, q, prod))

    # Sort by gain descending for greedy selection
    merges.sort(reverse=True)

    # Greedy selection of disjoint merges
    used_primes = set()
    selected_merges = []

    for gain, p, q, prod in merges:
        if p not in used_primes and q not in used_primes:
            selected_merges.append((p, q, prod))
            used_primes.add(p)
            used_primes.add(q)

    # Construct final result
    result = [1]
    for p in primes:
        if p not in used_primes:
            result.append(prime_powers[p])

    # Add the merged products
    for p, q, prod in selected_merges:
        result.append(prod)

    # Verify all constraints
    if len(set(result)) != len(result):
        # Fallback to prime powers only
        result = [1] + [prime_powers[p] for p in primes]

    for x in result:
        if not isinstance(x, int) or x < 1 or x > N:
            result = [1] + [prime_powers[p] for p in primes]
            break

    # Check pairwise coprime
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        result = [1] + [prime_powers[p] for p in primes]

    # Ensure we return Python ints in a list
    result = [int(x) for x in result]

    # Deadline check
    if time.time() > deadline:
        # Return what we have (should be valid by construction)
        pass

    return result

# EVOLVE-BLOCK-END
