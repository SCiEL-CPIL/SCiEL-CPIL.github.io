"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import heapq
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    limit = 10**18
    
    # Generate enough primes
    def gen_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    primes = gen_primes(200)
    
    def divisors_of_squarefree(prime_set):
        """Generate all divisors of a squarefree number given its prime set."""
        k = len(prime_set)
        divs = [1]
        for p in prime_set:
            divs = divs + [d * p for d in divs]
        divs.sort()
        return divs
    
    def tau_perp(n, prime_set=None):
        """Compute tau_perp: number of adjacent coprime pairs in sorted divisors."""
        if prime_set is None:
            # Factor n to get prime set
            m = n
            ps = []
            for p in primes:
                if p * p > m:
                    break
                if m % p == 0:
                    ps.append(p)
                    m //= p
            if m > 1:
                ps.append(m)
            prime_set = ps
        
        divs = divisors_of_squarefree(prime_set)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    # Find best k: product of smallest k primes <= 10^18
    best_k = 0
    prod = 1
    for i, p in enumerate(primes):
        if prod * p > limit:
            break
        prod *= p
        best_k = i + 1
    
    # Start with smallest k primes
    best_set = list(primes[:best_k])
    best_score = tau_perp(1, best_set)
    for p in best_set:
        pass  # just to trigger computation
    n_init = 1
    for p in best_set:
        n_init *= p
    best_score = tau_perp(n_init, best_set)
    
    # Local search: try single swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            current_p = best_set[i]
            # Try replacing with a larger prime
            for j in range(len(primes)):
                if time.time() > deadline:
                    break
                new_p = primes[j]
                if new_p == current_p:
                    continue
                # Check if new_p is already in the set
                if new_p in best_set:
                    continue
                # Check product constraint
                new_prod = n_init // current_p * new_p
                if new_prod > limit:
                    continue
                new_set = best_set[:]
                new_set[i] = new_p
                new_set.sort()
                s = tau_perp(new_prod, new_set)
                if s > best_score:
                    best_score = s
                    best_set = new_set
                    n_init = new_prod
                    improved = True
                    break  # restart from this improvement
    
    # Try double swaps if time permits
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(best_set)):
                    if time.time() > deadline:
                        break
                    p_i = best_set[i]
                    p_j = best_set[j]
                    for a in range(len(primes)):
                        if time.time() > deadline:
                            break
                        new_a = primes[a]
                        if new_a in best_set:
                            continue
                        for b in range(len(primes)):
                            if time.time() > deadline:
                                break
                            new_b = primes[b]
                            if new_b in best_set:
                                continue
                            if new_a == new_b:
                                continue
                            if new_a == p_i and new_b == p_j:
                                continue
                            if new_a == p_j and new_b == p_i:
                                continue
                            new_prod = n_init // (p_i * p_j) * new_a * new_b
                            if new_prod > limit:
                                continue
                            new_set = best_set[:]
                            new_set[i] = new_a
                            new_set[j] = new_b
                            new_set.sort()
                            s = tau_perp(new_prod, new_set)
                            if s > best_score:
                                best_score = s
                                best_set = new_set
                                n_init = new_prod
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return n_init
# EVOLVE-BLOCK-END
