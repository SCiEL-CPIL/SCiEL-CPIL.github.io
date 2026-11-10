import math
import time

def _sieve(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _get_divisors(prime_set):
    """Get all divisors of a squarefree number given its prime factors."""
    divs = [1]
    for p in prime_set:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * p)
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(n, prime_set):
    """Count adjacent coprime pairs in sorted divisors of squarefree n."""
    divs = _get_divisors(prime_set)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _score_from_primes(primes):
    """Compute tau_perp for a set of primes (squarefree number)."""
    if not primes:
        return 0
    divs = [1]
    for p in primes:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * p)
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _product(primes):
    """Compute product of primes."""
    prod = 1
    for p in primes:
        prod *= p
    return prod

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Find all primes up to a reasonable limit
    max_prime_limit = 500
    all_primes = _sieve(max_prime_limit)
    
    # Determine how many of the smallest primes we can fit
    prod = 1
    max_k = 0
    for i, p in enumerate(all_primes):
        if prod * p <= 10**18:
            prod *= p
            max_k = i + 1
        else:
            break
    
    # Initial solution: first max_k primes
    initial_set = all_primes[:max_k]
    best_score = _score_from_primes(initial_set)
    best_n = _product(initial_set)
    best_set = list(initial_set)
    
    # Local search: try swapping one prime out and one prime in
    # Use more aggressive search with multiple passes
    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 10.0:
        improved = False
        pass_count += 1
        for i in range(len(best_set)):
            if time.time() > deadline - 10.0:
                break
            for j in range(len(all_primes)):
                if time.time() > deadline - 10.0:
                    break
                # Skip if same prime or already in set
                if all_primes[j] == best_set[i]:
                    continue
                if all_primes[j] in best_set:
                    continue
                
                # Try swap
                new_set = list(best_set)
                new_set[i] = all_primes[j]
                
                # Check product
                new_prod = 1
                for p in new_set:
                    new_prod *= p
                    if new_prod > 10**18:
                        break
                if new_prod > 10**18:
                    continue
                
                new_score = _score_from_primes(new_set)
                if new_score > best_score:
                    best_score = new_score
                    best_n = new_prod
                    best_set = new_set
                    improved = True
    
    # Also try adding a prime if possible (increase k)
    if time.time() < deadline - 10.0:
        for j in range(len(all_primes)):
            if time.time() > deadline - 10.0:
                break
            if all_primes[j] in best_set:
                continue
            new_set = list(best_set) + [all_primes[j]]
            new_prod = best_n * all_primes[j]
            if new_prod > 10**18:
                continue
            new_score = _score_from_primes(new_set)
            if new_score > best_score:
                best_score = new_score
                best_n = new_prod
                best_set = new_set
    
    # Try removing a prime if it improves score (unlikely but check)
    if time.time() < deadline - 10.0:
        for i in range(len(best_set)):
            if time.time() > deadline - 10.0:
                break
            new_set = list(best_set)
            new_set.pop(i)
            new_prod = _product(new_set)
            new_score = _score_from_primes(new_set)
            if new_score > best_score:
                best_score = new_score
                best_n = new_prod
                best_set = new_set
    
    return best_n
