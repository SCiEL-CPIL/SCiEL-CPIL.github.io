import math
import time
from itertools import combinations

def _divisors_bitmasks(primes):
    """Generate divisors as (value, bitmask) pairs for a squarefree number.
    Bitmask has bit i set if prime i divides the divisor."""
    k = len(primes)
    divs = [(1, 0)]
    for i, p in enumerate(primes):
        bit = 1 << i
        new_divs = [(v * p, m | bit) for v, m in divs]
        divs.extend(new_divs)
    divs.sort(key=lambda x: x[0])
    return divs

def _tau_perp_bitmask(divs):
    """Count adjacent coprime pairs using bitmask representation.
    Two divisors are coprime iff their bitmasks are disjoint (AND == 0)."""
    count = 0
    for i in range(len(divs) - 1):
        if (divs[i][1] & divs[i + 1][1]) == 0:
            count += 1
    return count

def _score_primes(prime_list):
    """Score a list of primes as a squarefree number using bitmask method."""
    n = 1
    for p in prime_list:
        n *= p
        if n > 10**18:
            return -1, n
    divs = _divisors_bitmasks(prime_list)
    return _tau_perp_bitmask(divs), n

def _is_prime(n):
    """Check if n is prime."""
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

def _generate_primes(limit):
    """Generate all primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _greedy_local_search(prime_set, all_candidates, deadline):
    """Greedy local search: try swapping in/out primes to improve score."""
    current_score, current_n = _score_primes(prime_set)
    if current_n > 10**18:
        return prime_set, current_score, current_n
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap_score = current_score
        best_swap_set = prime_set
        
        # Try adding a prime not in the set
        for p in all_candidates:
            if p in prime_set:
                continue
            new_set = prime_set + [p]
            new_n = current_n * p
            if new_n > 10**18:
                continue
            s, _ = _score_primes(new_set)
            if s > best_swap_score:
                best_swap_score = s
                best_swap_set = new_set
                improved = True
        
        # Try removing a prime from the set (if size > 2)
        if len(prime_set) > 2:
            for i in range(len(prime_set)):
                new_set = prime_set[:i] + prime_set[i+1:]
                new_n = current_n // prime_set[i]
                s, _ = _score_primes(new_set)
                if s > best_swap_score:
                    best_swap_score = s
                    best_swap_set = new_set
                    improved = True
        
        # Try swapping one prime for another
        for i in range(len(prime_set)):
            for p in all_candidates:
                if p in prime_set:
                    continue
                new_set = prime_set[:i] + [p] + prime_set[i+1:]
                new_n = current_n // prime_set[i] * p
                if new_n > 10**18:
                    continue
                s, _ = _score_primes(new_set)
                if s > best_swap_score:
                    best_swap_score = s
                    best_swap_set = new_set
                    improved = True
        
        if improved:
            prime_set = best_swap_set
            current_score = best_swap_score
            current_n = 1
            for p in prime_set:
                current_n *= p
    
    return prime_set, current_score, current_n

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate primes up to 200 for a larger search space
    all_primes = _generate_primes(200)
    
    best_score = -1
    best_n = 2
    
    def try_candidate(prime_list):
        nonlocal best_score, best_n
        if time.time() > deadline:
            return False
        s, n = _score_primes(prime_list)
        if n <= 10**18 and s > best_score:
            best_score = s
            best_n = n
        return True
    
    # Phase 1: Try primorials of first k primes for k=2..15
    for k in range(2, 16):
        if time.time() > deadline - 30:
            break
        primes_k = all_primes[:k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
            if n_prod > 10**18:
                break
        else:
            try_candidate(primes_k)
            continue
        break
    
    # Phase 2: Systematic single swap search from primorial bases
    for base_size in range(12, 16):
        if time.time() > deadline - 25:
            break
        base = all_primes[:base_size]
        base_n = 1
        for p in base:
            base_n *= p
        if base_n > 10**18:
            continue
        
        used = set(base)
        for i in range(base_size):
            if time.time() > deadline - 20:
                break
            for j in range(base_size, len(all_primes)):
                if all_primes[j] in used:
                    continue
                new_primes = base[:i] + [all_primes[j]] + base[i+1:]
                new_n = base_n // base[i] * all_primes[j]
                if new_n > 10**18:
                    continue
                try_candidate(new_primes)
                if time.time() > deadline - 15:
                    break
    
    # Phase 3: Greedy local search from multiple starting points
    if time.time() < deadline - 40 and best_score > 0:
        # Factor best_n to get its primes
        best_n_temp = best_n
        best_primes = []
        for p in all_primes:
            if best_n_temp % p == 0:
                best_primes.append(p)
                best_n_temp //= p
        
        if len(best_primes) > 0 and best_n_temp == 1:
            # Run greedy local search from the best candidate
            refined_set, refined_score, refined_n = _greedy_local_search(
                best_primes, all_primes, deadline - 30
            )
            if refined_n <= 10**18 and refined_score > best_score:
                best_score = refined_score
                best_n = refined_n
        
        # Also try greedy from other good starting points
        for k in [13, 14]:
            if time.time() > deadline - 35:
                break
            base = all_primes[:k]
            base_n = 1
            for p in base:
                base_n *= p
            if base_n > 10**18:
                continue
            refined_set, refined_score, refined_n = _greedy_local_search(
                base, all_primes, deadline - 30
            )
            if refined_n <= 10**18 and refined_score > best_score:
                best_score = refined_score
                best_n = refined_n
    
    # Phase 4: Double swap search for best base sizes
    for base_size in [14, 15]:
        if time.time() > deadline - 20:
            break
        base = all_primes[:base_size]
        base_n = 1
        for p in base:
            base_n *= p
        if base_n > 10**18:
            continue
        
        used = set(base)
        for i in range(base_size):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, base_size):
                if time.time() > deadline - 15:
                    break
                for k in range(base_size, min(len(all_primes), base_size + 15)):
                    if time.time() > deadline - 15:
                        break
                    for l in range(k + 1, min(len(all_primes), base_size + 15)):
                        if all_primes[k] in used or all_primes[l] in used:
                            continue
                        new_primes = base[:i] + [all_primes[k]] + base[i+1:j] + [all_primes[l]] + base[j+1:]
                        new_n = base_n // base[i] // base[j] * all_primes[k] * all_primes[l]
                        if new_n > 10**18:
                            continue
                        try_candidate(new_primes)
                        if time.time() > deadline - 10:
                            break
    
    # Phase 5: Final greedy polish from the best candidate
    if time.time() < deadline - 10 and best_score > 0:
        best_n_temp = best_n
        best_primes = []
        for p in all_primes:
            if best_n_temp % p == 0:
                best_primes.append(p)
                best_n_temp //= p
        
        if len(best_primes) > 0 and best_n_temp == 1:
            refined_set, refined_score, refined_n = _greedy_local_search(
                best_primes, all_primes, deadline - 5
            )
            if refined_n <= 10**18 and refined_score > best_score:
                best_score = refined_score
                best_n = refined_n
    
    return best_n
