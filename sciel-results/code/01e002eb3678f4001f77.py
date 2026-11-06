"""Combined program: Maximize coprime adjacent divisor pairs using bitmask-based local search."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _primes_up_to(limit):
    """Generate primes up to limit using sieve of Eratosthenes."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

# Prime pool for search
_PRIMES = _primes_up_to(500)

def _tau_perp_bitmask(primes, exponents):
    """
    Compute tau_perp score using bitmask representation for speed.
    For each divisor, we track which primes divide it as a bitmask.
    Two adjacent divisors are coprime iff their bitmasks have no common bits.
    """
    # Generate all divisors with their prime factor bitmasks
    # Each divisor is represented as (value, bitmask)
    # bitmask bit i is set if primes[i] divides the divisor
    
    divisors = [(1, 0)]
    
    for idx, (p, e) in enumerate(zip(primes, exponents)):
        if e == 0:
            continue
        bit = 1 << idx
        current = list(divisors)
        pk = 1
        for k in range(1, e + 1):
            pk *= p
            for d_val, d_mask in current:
                divisors.append((d_val * pk, d_mask | bit))
    
    # Sort by value
    divisors.sort(key=lambda x: x[0])
    
    if len(divisors) < 2:
        return 0
    
    count = 0
    for i in range(len(divisors) - 1):
        # Two divisors are coprime iff their prime factor bitmasks are disjoint
        if (divisors[i][1] & divisors[i+1][1]) == 0:
            count += 1
    return count

def _compute_n(primes, exponents):
    """Compute n from prime factorization."""
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def construct():
    """
    Returns an integer n <= 10^18 that maximizes the number of adjacent 
    coprime pairs in its sorted list of divisors.
    
    Uses bitmask-based score computation for speed, combined with local search
    including exponent changes and prime replacement moves.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    max_n = 10**18
    
    best_score = 0
    best_n = 2
    
    # Generate candidate base states: primorials of first k primes
    bases = []
    prod = 1
    for i, p in enumerate(_PRIMES):
        if prod * p > max_n:
            break
        prod *= p
        bases.append((i + 1, prod))
    
    # Search from the top few bases (largest primorials first)
    # P15, P14, P13 are the largest
    search_bases = bases[-4:]  # Last 4 in the list
    
    for k, base_prod in search_bases:
        if time.time() > deadline:
            break
            
        # Initial state: first k primes, exponent 1 each
        current_primes = _PRIMES[:k]
        current_exps = [1] * k
        current_n = base_prod
        current_score = _tau_perp_bitmask(current_primes, current_exps)
        
        if current_score > best_score:
            best_score = current_score
            best_n = current_n
        
        # Local Search
        improved = True
        iterations = 0
        max_iterations = 500  # Safety cap per base
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Move 1: Increase exponent of prime i
            for i in range(k):
                if time.time() > deadline:
                    break
                new_exps = current_exps[:]
                new_exps[i] += 1
                new_n = current_n * current_primes[i]
                if new_n <= max_n:
                    score = _tau_perp_bitmask(current_primes, new_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
            
            # Move 2: Decrease exponent of prime i (if > 1)
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 1:
                    continue
                new_exps = current_exps[:]
                new_exps[i] -= 1
                new_n = current_n // current_primes[i]
                if new_n >= 2:
                    score = _tau_perp_bitmask(current_primes, new_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
            
            # Move 3: Swap exponents between primes i and j
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 0:
                    continue
                for j in range(k):
                    if i == j:
                        continue
                    new_exps = current_exps[:]
                    new_exps[i] -= 1
                    new_exps[j] += 1
                    new_n = (current_n // current_primes[i]) * current_primes[j]
                    if new_n <= max_n:
                        score = _tau_perp_bitmask(current_primes, new_exps)
                        if score > current_score:
                            current_exps = new_exps
                            current_n = new_n
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
            
            # Move 4: Replace prime at position i with a different prime from the pool
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 0:
                    continue
                old_p = current_primes[i]
                # Try replacing with primes from the pool not already in the set
                for j in range(len(_PRIMES)):
                    if time.time() > deadline:
                        break
                    new_p = _PRIMES[j]
                    if new_p == old_p:
                        continue
                    if new_p in current_primes:
                        continue
                    new_primes = current_primes[:]
                    new_primes[i] = new_p
                    new_n = (current_n // old_p) * new_p
                    if new_n <= max_n:
                        score = _tau_perp_bitmask(new_primes, current_exps)
                        if score > current_score:
                            current_primes = new_primes
                            current_n = new_n
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                            break  # Take the first improvement for this position
        
        # After local search from this base, check time
        if time.time() > deadline:
            break
    
    # Final verification: ensure best_n is valid
    if best_n < 2:
        best_n = 2
    
    return best_n
# EVOLVE-BLOCK-END
