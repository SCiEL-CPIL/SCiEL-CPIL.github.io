import math
import random
import time

def _tau_perp_from_factors(factors):
    """
    Calculate tau_perp(n) given the prime factorization of n.
    factors: dict {prime: exponent}
    """
    # Generate divisors
    divisors = [1]
    for p, e in factors.items():
        new_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= p
        divisors = new_divisors
        
    if len(divisors) < 2:
        return 0

    divisors.sort()
    
    score = 0
    prev = divisors[0]
    for i in range(1, len(divisors)):
        curr = divisors[i]
        if math.gcd(prev, curr) == 1:
            score += 1
        prev = curr
        
    return score

def _is_valid(factors, limit=10**18):
    """Check if the product of primes^exponents is <= limit."""
    n = 1
    for p, e in factors.items():
        n *= p ** e
        if n > limit:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Small primes for our search space
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    # Initial candidates
    # 1. Primorial of 15 primes
    p15 = 1
    for p in primes[:15]:
        p15 *= p
    cand1 = {p: 1 for p in primes[:15]}
    
    # 2. A highly composite-like number
    # 2^a * 3^b * 5^c ...
    # Let's try a known good one or generate a few
    candidates = [cand1]
    
    # Add some random initial candidates to diversify
    for _ in range(10):
        factors = {}
        # Random exponents, decreasing
        max_e = random.randint(1, 20)
        for i, p in enumerate(primes[:15]):
            if max_e == 0: break
            # Ensure decreasing
            e = random.randint(0, max_e)
            if e > 0:
                factors[p] = e
            # Update max_e for next
            if e > 0:
                max_e = e
            else:
                break # If we hit 0, subsequent must be 0 due to decreasing constraint? 
                # Actually, we can have gaps? No, for optimal structure, 
                # if e_i = 0, then e_{i+1} must be 0 because p_{i+1} > p_i.
                # So we stop.
        if factors and _is_valid(factors):
            candidates.append(factors)
            
    best_factors = candidates[0]
    best_score = _tau_perp_from_factors(best_factors)
    
    # Local Search
    current_factors = dict(best_factors)
    
    while time.time() < deadline:
        # Mutation
        new_factors = dict(current_factors)
        
        # Pick a random prime from the list (or from current factors)
        # Strategy:
        # 1. Increase exponent of a small prime (if valid)
        # 2. Decrease exponent of a large prime (to free up budget)
        # 3. Swap: decrease large, increase small
        
        action = random.random()
        
        # Get list of primes in current factors
        active_primes = [p for p in primes if p in new_factors]
        
        if not active_primes:
            # Start fresh
            current_factors = {primes[0]: 1}
            continue

        if action < 0.4:
            # Increase exponent of a random active prime
            p = random.choice(active_primes)
            new_factors[p] += 1
        elif action < 0.8:
            # Decrease exponent of a random active prime
            p = random.choice(active_primes)
            if new_factors[p] > 1:
                new_factors[p] -= 1
            else:
                del new_factors[p]
        else:
            # Try to add a new prime or remove one
            # Add next unused prime
            used_primes = set(new_factors.keys())
            available_primes = [p for p in primes if p not in used_primes]
            if available_primes and random.random() < 0.5:
                # Add smallest available
                p = available_primes[0]
                new_factors[p] = 1
            elif active_primes:
                # Remove a random active prime
                p = random.choice(active_primes)
                del new_factors[p]
        
        # Check validity
        if not _is_valid(new_factors):
            # If invalid, try to repair by reducing exponents of largest primes
            # Or just reject
            continue
            
        # Check structure: e1 >= e2 >= ...
        # This is a heuristic constraint. Optimal numbers usually satisfy this.
        # Let's enforce it by sorting exponents and re-assigning to smallest primes?
        # No, the primes are fixed. We just check if the current state violates it?
        # Actually, if we mutate, we might violate e_i >= e_{i+1}.
        # We should "normalize" the candidate:
        # Sort primes, sort exponents in descending order, and pair them.
        # This ensures we are always looking at a "canonical" form.
        
        # Normalize
        if new_factors:
            sorted_primes = sorted(new_factors.keys())
            sorted_exponents = sorted(new_factors.values(), reverse=True)
            # Re-pair
            normalized = {}
            for p, e in zip(sorted_primes, sorted_exponents):
                if e > 0:
                    normalized[p] = e
            new_factors = normalized
        
        # Re-check validity after normalization (product is the same, so still valid)
        
        score = _tau_perp_from_factors(new_factors)
        
        if score > best_score:
            best_score = score
            best_factors = new_factors
            current_factors = new_factors
        else:
            # Accept with some probability to escape local minima (Simulated Annealing style)
            # Simple: accept if random < 0.1
            if random.random() < 0.1:
                current_factors = new_factors
            else:
                # Occasionally restart from best
                if random.random() < 0.05:
                    current_factors = dict(best_factors)
                    
    # Convert best_factors to integer
    n = 1
    for p, e in best_factors.items():
        n *= p ** e
        
    return n
