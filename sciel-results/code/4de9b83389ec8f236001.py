import math
import time
from itertools import combinations

MAX_N = 10**18
DEADLINE = 140.0 # Safety margin for 150s limit

# First 20 primes
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

def get_divisors(prime_powers):
    """
    prime_powers: list of (prime, exponent)
    Returns sorted list of divisors
    """
    divs = [1]
    for p, e in prime_powers:
        new_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * power)
            power *= p
        divs = new_divs
    divs.sort()
    return divs

def score(n, divs=None):
    if divs is None:
        # Factorize n? No, we should pass divs to be fast.
        # But for the start, we need a way to generate them from n.
        # We'll stick to passing prime_powers.
        pass
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def evaluate(prime_powers):
    """
    Returns (score, n) for a given list of (prime, exponent)
    """
    n = 1
    for p, e in prime_powers:
        n *= (p ** e)
        if n > MAX_N:
            return -1, n
    divs = get_divisors(prime_powers)
    s = score(n, divs)
    return s, n

def construct():
    start_time = time.time()
    
    # Initial best: Squarefree with smallest 15 primes
    # Product of first 15 primes is ~6e17
    init_primes = PRIMES[:15]
    init_pp = [(p, 1) for p in init_primes]
    best_score, best_n = evaluate(init_pp)
    
    # List of candidate prime powers to try
    # We focus on the first ~16 primes
    # We'll do a local search by modifying exponents
    
    # We represent the state as a list of exponents for PRIMES[:20]
    # But most will be 0 or 1.
    
    # Let's try a few specific "dense" configurations
    candidates = []
    
    # 1. Squarefree, 15 primes (current best likely)
    candidates.append([(p, 1) for p in PRIMES[:15]])
    
    # 2. Try increasing exponent of 2
    # 2^2 * 3 * ... * 43 (14 primes)
    candidates.append([(2, 2)] + [(p, 1) for p in PRIMES[1:15]])
    # 2^3 * 3 * ... * 41 (13 primes)
    candidates.append([(2, 3)] + [(p, 1) for p in PRIMES[1:14]])
    # 2^4 * 3 * ... * 37 (12 primes)
    candidates.append([(2, 4)] + [(p, 1) for p in PRIMES[1:13]])
    # 2^5 * 3 * ... * 31 (11 primes)
    candidates.append([(2, 5)] + [(p, 1) for p in PRIMES[1:12]])
    
    # 3. Try 3^2
    # 2 * 3^2 * 5 * ... * 43
    candidates.append([(2, 1), (3, 2)] + [(p, 1) for p in PRIMES[2:15]])
    # 2 * 3^2 * 5 * ... * 41
    candidates.append([(2, 1), (3, 2)] + [(p, 1) for p in PRIMES[2:14]])
    
    # 4. Try 2^2 * 3^2
    # 2^2 * 3^2 * 5 * ... * 41
    candidates.append([(2, 2), (3, 2)] + [(p, 1) for p in PRIMES[2:14]])
    # 2^2 * 3^2 * 5 * ... * 37
    candidates.append([(2, 2), (3, 2)] + [(p, 1) for p in PRIMES[2:13]])
    
    # 5. Try 2^2 * 3 * 5^2
    candidates.append([(2, 2), (3, 1), (5, 2)] + [(p, 1) for p in PRIMES[3:15]])
    
    # 6. General heuristic search
    # Start with the best found so far, try to improve
    # We'll just evaluate all candidates first
    for pp in candidates:
        s, n = evaluate(pp)
        if s > best_score:
            best_score = s
            best_n = n
        if time.time() - start_time > DEADLINE:
            break
            
    # If we have time, do a small local search from the best candidate
    # Find the pp that gave best_score
    best_pp = None
    for pp in candidates:
        s, n = evaluate(pp)
        if s == best_score and n == best_n:
            best_pp = pp
            break
            
    if best_pp and time.time() - start_time < DEADLINE - 10:
        # Convert to dict for easier manipulation? Or just list.
        # We'll try increasing exponents of small primes in best_pp
        # and decreasing exponents of large primes (removing them)
        
        # Create a working copy
        current_pp = best_pp[:]
        
        # Try all single step moves
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            # Try increasing exponent of each prime
            for i in range(len(current_pp)):
                p, e = current_pp[i]
                new_pp = current_pp[:]
                new_pp[i] = (p, e + 1)
                s, n = evaluate(new_pp)
                if s > best_score:
                    best_score = s
                    best_n = n
                    best_pp = new_pp
                    current_pp = new_pp
                    improved = True
                    break # Restart search from new best
                    
            # Try removing a prime (setting exponent to 0) if e>0
            # Actually, if we remove a prime, we might be able to increase another.
            # But for now, just increasing exponents is the main path to more divisors.
            # Note: We are limited by MAX_N.
            
    return best_n

construct()
