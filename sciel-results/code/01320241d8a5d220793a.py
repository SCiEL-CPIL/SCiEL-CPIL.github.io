import math
import time
from itertools import combinations

def _tau_perp_squarefree(primes):
    """
    Count adjacent coprime pairs among divisors of a squarefree number
    using a Gray-code traversal of the Boolean hypercube.
    
    For a squarefree number with k prime factors, divisors correspond to
    subsets of {0, ..., k-1}. We traverse in Gray code order, but we need
    the sorted order of divisors. Instead, we generate all divisors, sort them,
    and count. But for efficiency with the Gray code insight:
    
    Actually, the key insight is that we can compute the sorted divisors
    more efficiently. For k <= 15, 2^k <= 32768, so generating and sorting
    is fast. The Gray code insight helps when we want to avoid the sort.
    
    Let me use a different approach: generate divisors in sorted order using
    a heap or by leveraging the structure.
    
    Actually for k up to 15, just generate all 2^k divisors and sort.
    """
    k = len(primes)
    divs = [1]
    for p in primes:
        new_divs = [d * p for d in divs]
        divs.extend(new_divs)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _score_primes(prime_list):
    """Score a list of primes as a squarefree number."""
    n = 1
    for p in prime_list:
        n *= p
    if n > 10**18:
        return -1, n
    return _tau_perp_squarefree(prime_list), n

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate primes up to ~200 for more search space
    def _sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    all_primes = _sieve(250)
    
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
    
    # Phase 1: Try primorials of first k primes for k=2..16
    for k in range(2, 17):
        if k >= len(all_primes):
            break
        primes_k = all_primes[:k]
        n_prod = 1
        for p in primes_k:
            n_prod *= p
        if n_prod > 10**18:
            break
        try_candidate(primes_k)
    
    # Phase 2: Single swap - replace one prime in a base set with a larger one
    for base_size in [14, 15, 16]:
        if base_size > len(all_primes):
            continue
        base = all_primes[:base_size]
        base_n = 1
        for p in base:
            base_n *= p
        if base_n > 10**18:
            continue
        
        used = set(base)
        for i in range(base_size):
            for j in range(base_size, len(all_primes)):
                if time.time() > deadline:
                    return best_n
                if all_primes[j] in used:
                    continue
                new_n = base_n // base[i] * all_primes[j]
                if new_n > 10**18:
                    continue
                candidate = base[:i] + [all_primes[j]] + base[i+1:]
                candidate.sort()
                try_candidate(candidate)
    
    # Phase 3: Double swap - replace two primes
    for base_size in [14, 15]:
        if base_size > len(all_primes):
            continue
        base = all_primes[:base_size]
        base_n = 1
        for p in base:
            base_n *= p
        if base_n > 10**18:
            continue
        
        for i in range(base_size):
            for j in range(i + 1, base_size):
                for k in range(base_size, len(all_primes)):
                    for l in range(k + 1, len(all_primes)):
                        if time.time() > deadline:
                            return best_n
                        new_n = base_n // base[i] // base[j] * all_primes[k] * all_primes[l]
                        if new_n > 10**18:
                            continue
                        candidate = [p for idx, p in enumerate(base) if idx != i and idx != j]
                        candidate.append(all_primes[k])
                        candidate.append(all_primes[l])
                        candidate.sort()
                        try_candidate(candidate)
    
    # Phase 4: Local search - perturb best solution
    def factorize(n):
        factors = []
        temp = n
        for p in all_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.append(temp)
        return factors
    
    if best_n > 2:
        best_primes = factorize(best_n)
        if len(best_primes) <= 16:
            used = set(best_primes)
            for i in range(len(best_primes)):
                for j in range(len(all_primes)):
                    if time.time() > deadline:
                        return best_n
                    if all_primes[j] in used:
                        continue
                    new_n = best_n // best_primes[i] * all_primes[j]
                    if new_n > 10**18:
                        continue
                    candidate = best_primes[:i] + [all_primes[j]] + best_primes[i+1:]
                    candidate.sort()
                    try_candidate(candidate)
    
    # Phase 5: Try non-consecutive prime sets - skip some small primes
    # This explores the space more broadly
    if time.time() < deadline:
        # Try sets of 14-16 primes that aren't just the first k
        for target_size in [14, 15, 16]:
            if target_size > len(all_primes):
                continue
            # Try removing the largest prime from first target_size+1 and adding others
            base = all_primes[:target_size + 1]
            base_n = 1
            for p in base:
                base_n *= p
            
            # Remove one prime and try to add a different one
            for remove_idx in range(target_size + 1):
                remaining = [p for idx, p in enumerate(base) if idx != remove_idx]
                rem_n = 1
                for p in remaining:
                    rem_n *= p
                
                for add_p in all_primes:
                    if time.time() > deadline:
                        return best_n
                    if add_p in remaining:
                        continue
                    new_n = rem_n * add_p
                    if new_n > 10**18:
                        continue
                    candidate = remaining + [add_p]
                    candidate.sort()
                    try_candidate(candidate)
    
    # Phase 6: Random sampling of prime sets
    if time.time() < deadline:
        import random
        random.seed(42)
        for _ in range(50000):
            if time.time() > deadline:
                break
            # Randomly pick 14-16 primes
            target_size = random.choice([14, 15, 16])
            if target_size > len(all_primes):
                target_size = len(all_primes)
            candidate_primes = random.sample(all_primes, min(target_size, len(all_primes)))
            n_prod = 1
            for p in candidate_primes:
                n_prod *= p
            if n_prod > 10**18:
                continue
            candidate_primes.sort()
            try_candidate(candidate_primes)
    
    return best_n
