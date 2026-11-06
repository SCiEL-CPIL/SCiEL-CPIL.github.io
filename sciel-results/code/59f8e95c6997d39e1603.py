import math
import time
from math import gcd

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

# We need primes up to at least 100 to allow swapping in larger primes if needed
_PRIMES = _primes_up_to(300)

def _tau_perp(primes, exponents):
    """
    Compute tau_perp score from prime factorization.
    primes: list of prime bases
    exponents: list of exponents corresponding to primes
    """
    # Generate divisors
    divisors = [1]
    for p, e in zip(primes, exponents):
        if e == 0:
            continue
        current = list(divisors)
        pk = 1
        for k in range(1, e + 1):
            pk *= p
            for d in current:
                divisors.append(d * pk)
    
    divisors.sort()
    if len(divisors) < 2:
        return 0
    
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _compute_n(primes, exponents):
    n = 1
    for p, e in zip(primes, exponents):
        if e > 0:
            n *= p ** e
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    max_n = 10**18
    
    best_score = 0
    best_n = 2
    
    # Candidate bases: Primorials of the first k primes
    # P15 is the largest primorial <= 10^18
    # P14, P13 are smaller, allowing for higher exponents
    
    bases = []
    prod = 1
    for i, p in enumerate(_PRIMES):
        if prod * p > max_n:
            break
        prod *= p
        # Store the base state: first i+1 primes, all exp 1
        bases.append((i + 1, prod))
    
    # We will run local search from the top few bases
    # Sort bases by primorial size descending (largest first) or just take the top 3-4
    # P15, P14, P13
    search_bases = bases[-3:] # Last 3 in the list (largest primorials)
    
    for k, base_prod in search_bases:
        if time.time() > deadline:
            break
            
        # Initial state: first k primes, exponent 1
        current_primes = _PRIMES[:k]
        current_exps = [1] * k
        current_n = base_prod
        current_score = _tau_perp(current_primes, current_exps)
        
        if current_score > best_score:
            best_score = current_score
            best_n = current_n
            
        # Local Search
        # We allow modifying exponents of the first k primes.
        # We also allow swapping in a prime from _PRIMES[k:k+10] if we remove one.
        # For simplicity and speed, we restrict to modifying exponents of the current set
        # and swapping within a slightly larger pool.
        
        # To keep the state manageable, we fix the set of primes to the first k primes 
        # plus a few larger ones, but initially just use the first k.
        # If we remove a prime, we might want to add a different one.
        # Let's define the pool of primes we consider for indices 0..k-1.
        # We will allow the exponents to change.
        
        improved = True
        iterations = 0
        max_iterations = 800 # Safety cap per base
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Generate moves
            # 1. Increase exponent of prime i
            for i in range(k):
                if time.time() > deadline:
                    break
                new_exps = current_exps[:]
                new_exps[i] += 1
                new_n = current_n * current_primes[i]
                if new_n <= max_n:
                    score = _tau_perp(current_primes, new_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            
            # 2. Decrease exponent of prime i (if > 1)
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 1:
                    continue
                new_exps = current_exps[:]
                new_exps[i] -= 1
                new_n = current_n // current_primes[i]
                if new_n >= 2:
                    score = _tau_perp(current_primes, new_exps)
                    if score > current_score:
                        current_exps = new_exps
                        current_n = new_n
                        current_score = score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n

            # 3. Swap: Decrease exponent of prime i, Increase exponent of prime j
            # This is useful if we want to shift "mass" from a large prime to a small one
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
                        score = _tau_perp(current_primes, new_exps)
                        if score > current_score:
                            current_exps = new_exps
                            current_n = new_n
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n

            # 4. Replace prime: swap prime at position i with a different prime from the pool
            # This allows changing the set of primes, not just their exponents
            # We try replacing each prime with candidates from _PRIMES that are not already in the set
            for i in range(k):
                if time.time() > deadline:
                    break
                if current_exps[i] <= 0:
                    continue
                # Candidate replacement primes: those in _PRIMES not currently in current_primes
                # We limit to primes within a reasonable range to keep search tractable
                old_p = current_primes[i]
                for j in range(len(_PRIMES)):
                    if time.time() > deadline:
                        break
                    new_p = _PRIMES[j]
                    if new_p == old_p:
                        continue
                    if new_p in current_primes:
                        continue
                    # Try replacing prime i with new_p
                    new_primes = current_primes[:]
                    new_primes[i] = new_p
                    new_n = (current_n // old_p) * new_p
                    if new_n <= max_n:
                        score = _tau_perp(new_primes, current_exps)
                        if score > current_score:
                            current_primes = new_primes
                            current_n = new_n
                            current_score = score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n

    return best_n
