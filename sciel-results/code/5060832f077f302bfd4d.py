import math
import time
from math import gcd
from itertools import product

def _get_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

_PRIMES = _get_primes(1000)

def _tau_perp_fast(n, factorization):
    """
    Compute tau_perp(n) using the factorization.
    factorization: list of (prime, exponent)
    Returns the count of adjacent pairs of divisors with gcd 1.
    """
    # Generate all divisors
    divs = [1]
    for p, e in factorization:
        new_divs = []
        for d in divs:
            val = d
            for exp in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def _search():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    LIMIT = 10**18
    
    # Initial best: primorial of first 15 primes
    # 2*3*5*...*47 ~ 6.1e17
    # 2*3*...*53 ~ 3.2e19 > 10^18
    # So max squarefree k is 15.
    initial_primes = _PRIMES[:15]
    best_n = 1
    best_factorization = []
    for p in initial_primes:
        if best_n * p > LIMIT:
            break
        best_n *= p
        best_factorization.append((p, 1))
    
    best_score = _tau_perp_fast(best_n, best_factorization)
    
    # We will perform a local search on the exponents of the first K primes.
    # K is likely around 12-15.
    # We start with a good initial guess: exponents decreasing, e.g., [4, 2, 1, 1, ...] or similar.
    # Actually, for squarefree, score is high. Let's explore deviations from squarefree.
    
    # Define a function to calculate score for a given exponent vector
    # exponents: list of exponents for _PRIMES[0], _PRIMES[1], ...
    def evaluate_exponents(exponents):
        n = 1
        factorization = []
        for i, e in enumerate(exponents):
            if e == 0:
                continue
            p = _PRIMES[i]
            pe = p ** e
            if n * pe > LIMIT:
                return None, None, 0
            n *= pe
            factorization.append((p, e))
        
        if len(factorization) < 2:
            return None, None, 0
            
        # Optimization: if number of divisors is too large, skip detailed check?
        # 200k divisors is fast enough for gcd check.
        num_divs = 1
        for _, e in factorization:
            num_divs *= (e + 1)
        if num_divs > 500000:
            return None, None, 0

        score = _tau_perp_fast(n, factorization)
        return n, factorization, score

    # Start with the best squarefree candidate
    # Find max k such that primorial <= LIMIT
    max_k = 0
    prod = 1
    for i, p in enumerate(_PRIMES):
        if prod * p <= LIMIT:
            prod *= p
            max_k = i + 1
        else:
            break
            
    # Initial exponents: all 1s for first max_k primes
    current_exponents = [1] * max_k
    # Pad with zeros if necessary for safety in mutation
    current_exponents += [0] * (20 - max_k)
    
    current_n, current_fact, current_score = evaluate_exponents(current_exponents)
    if current_score > best_score:
        best_score = current_score
        best_n = current_n
        best_factorization = current_fact
        
    # Hill Climbing
    # We try to increase/decrease exponents of primes in the list.
    # We only care about the first ~18 primes.
    MAX_PRIMES_TO_CONSIDER = 18
    
    while time.time() < deadline - 5:
        improved = False
        # Try increasing exponents
        for i in range(MAX_PRIMES_TO_CONSIDER):
            if time.time() > deadline - 5:
                break
            if current_exponents[i] == 0:
                continue
            # Try increasing exponent of prime i
            new_exponents = current_exponents[:]
            new_exponents[i] += 1
            
            n, fact, score = evaluate_exponents(new_exponents)
            if score > best_score:
                best_score = score
                best_n = n
                best_factorization = fact
                current_exponents = new_exponents
                improved = True
                # If we improved, restart the loop to check other directions from new state
                break
        
        if improved:
            continue
            
        # Try decreasing exponents (to free up budget for other primes)
        for i in range(MAX_PRIMES_TO_CONSIDER):
            if time.time() > deadline - 5:
                break
            if current_exponents[i] <= 1:
                continue
            new_exponents = current_exponents[:]
            new_exponents[i] -= 1
            
            n, fact, score = evaluate_exponents(new_exponents)
            if score > best_score:
                best_score = score
                best_n = n
                best_factorization = fact
                current_exponents = new_exponents
                improved = True
                break
                
        if improved:
            continue
            
        # Try swapping: decrease one, increase another (if budget allows)
        # This is more complex. Let's try a simpler local move: 
        # Increase exponent of a smaller prime if we can decrease a larger one?
        # Or just try "shifting" budget.
        
        # Let's try a more aggressive move: try setting a prime to 0 and another to 2?
        # Actually, the hill climb above should find local maxima.
        # To escape local minima, we can try random perturbations if stuck.
        
        # Check if we are stuck for a while? 
        # Simple random restart from a different initial state if no improvement in last X seconds
        # For now, let's just add a random perturbation step if we haven't improved in a while.
        
        # We need a "stuck" counter or time tracker.
        # Since we don't have a global "last_improve_time" in the current structure easily,
        # let's just do a small random search if we complete a full pass without improvement.
        
        # To keep it simple and effective, let's just rely on the hill climb.
        # But hill climb can get stuck.
        
        # Let's add a phase: Random search in the neighborhood of the best solution found so far.
        
        # If we finish a pass without improvement, do a random mutation
        import random
        if not improved:
            # Random mutation
            # Pick a random prime index
            idx = random.randint(0, MAX_PRIMES_TO_CONSIDER - 1)
            if current_exponents[idx] > 0:
                new_exponents = current_exponents[:]
                # Random change: -1, 0, +1, +2
                delta = random.choice([-1, 0, 1, 1, 2])
                new_exponents[idx] = max(0, current_exponents[idx] + delta)
                
                n, fact, score = evaluate_exponents(new_exponents)
                if score > best_score:
                    best_score = score
                    best_n = n
                    best_factorization = fact
                    current_exponents = new_exponents
                    improved = True
                elif score == best_score:
                    # Accept neutral moves to explore
                    if random.random() < 0.1:
                        current_exponents = new_exponents

        if not improved and random.random() < 0.01:
             # Occasionally try a different starting point? No, stick to local search for speed.
             pass

    return best_n

def construct():
    return _search()
