import time
import math
import random

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

def get_divisors(primes):
    divs = [1]
    for p in primes:
        new_divs = [d * p for d in divs]
        divs.extend(new_divs)
    return divs

def score_from_primes(primes):
    divs = get_divisors(primes)
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count

def product(primes):
    p = 1
    for x in primes:
        p *= x
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate primes up to 200 for replacement candidates
    primes = sieve(200)
    
    # Start with the 15 smallest primes, which fit in 10^18
    # Product of first 15 primes is ~6.1e17
    base_primes = primes[:15]
    current_primes = base_primes[:]
    current_score = score_from_primes(current_primes)
    best_score = current_score
    best_primes = current_primes[:]
    
    # Local search: try to swap one prime with a larger one
    # to potentially improve the "interleaving" of divisors
    # We limit the number of evaluations to stay within time
    
    max_evals = 2000  # Rough estimate, will check time
    eval_count = 0
    
    # Precompute product for quick checks? 
    # current_prod = product(current_primes)
    
    used_set = set(current_primes)
    
    # Candidates for replacement: primes > max(current_primes) 
    # but we can also swap any prime with any other prime (including smaller) 
    # though smaller primes usually lead to denser divisor sets.
    # However, the "coprime adjacency" might be maximized by specific structures.
    
    # Strategy: Iterate through positions, try replacing with a random larger prime
    # that keeps product < 10^18
    
    while time.time() < deadline - 5.0:
        # Pick a random index to swap
        i = random.randint(0, len(current_primes) - 1)
        p_old = current_primes[i]
        
        # Find candidate replacements
        # We want to try primes that are not in used_set
        # To keep product small, we prefer smaller replacements, 
        # but to change structure, we might need larger ones.
        # Let's sample from primes up to 200.
        
        # Filter candidates: q not in used_set, and product(current_primes) / p_old * q <= 10^18
        current_prod = product(current_primes)
        max_allowed_q = int((10**18 * p_old) / current_prod)
        
        # Get primes in range [2, max_allowed_q] not in used_set
        # Since max_allowed_q can be large, we just check against our sieve list
        candidates = []
        for q in primes:
            if q in used_set:
                continue
            if q <= max_allowed_q:
                candidates.append(q)
        
        if not candidates:
            continue
            
        # Pick a random candidate
        q_new = random.choice(candidates)
        
        # Construct new set
        new_primes = current_primes[:]
        new_primes[i] = q_new
        
        # Check time before expensive score calc
        if time.time() > deadline - 10.0:
            break
            
        new_score = score_from_primes(new_primes)
        eval_count += 1
        
        if new_score > current_score:
            # Accept
            used_set.remove(p_old)
            used_set.add(q_new)
            current_primes = new_primes
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_primes = current_primes[:]
                
        # Optional: If we find a better score, we might want to reset the search focus
        # or continue. Simple hill climbing is fine.
        
    n = product(best_primes)
    if n > 10**18:
        # Fallback to base primes if something went wrong
        n = product(base_primes)
    return n
