"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _sieve(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _divisors_of_squarefree(prime_set):
    """Generate all divisors of a squarefree number from its prime factors."""
    divisors = [1]
    for p in prime_set:
        divisors += [d * p for d in divisors]
    return divisors

def _tau_perp_from_primes(prime_set):
    """Compute tau_perp for squarefree n given its prime factors."""
    divisors = _divisors_of_squarefree(prime_set)
    divisors.sort()
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _product_of_primes(prime_set):
    """Compute product of primes in the set."""
    prod = 1
    for p in prime_set:
        prod *= p
    return prod

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    limit = 10**18
    
    # Generate a larger prime pool
    max_prime_val = 2000
    primes = _sieve(max_prime_val)
    
    # Initial solution: first 15 primes
    k = 15
    current_primes = primes[:k]
    current_n = _product_of_primes(current_primes)
    current_score = _tau_perp_from_primes(current_primes)
    
    best_primes = list(current_primes)
    best_n = current_n
    best_score = current_score
    
    # Simulated annealing
    T_start = 50.0
    T_end = 0.1
    alpha = 0.9995  # cooling rate
    
    T = T_start
    step = 0
    
    while time.time() < deadline - 10:
        # Check if we should stop
        if T < T_end:
            break
        
        # Generate a random move
        move_type = random.random()
        new_primes = list(current_primes)
        
        if move_type < 0.6:
            # Single swap: replace one prime
            i = random.randint(0, len(new_primes) - 1)
            # Pick a random prime not in current set
            available = [p for idx, p in enumerate(primes) if p not in new_primes]
            if not available:
                T *= alpha
                continue
            new_p = random.choice(available)
            new_primes[i] = new_p
            new_primes.sort()
        elif move_type < 0.85:
            # Double swap: replace two primes
            if len(new_primes) < 2:
                T *= alpha
                continue
            i, j = random.sample(range(len(new_primes)), 2)
            available = [p for idx, p in enumerate(primes) if p not in new_primes]
            if len(available) < 2:
                T *= alpha
                continue
            new_p1, new_p2 = random.sample(available, 2)
            new_primes[i] = new_p1
            new_primes[j] = new_p2
            new_primes.sort()
        else:
            # Try adding or removing a prime (change k)
            if move_type < 0.92:
                # Add a prime
                available = [p for idx, p in enumerate(primes) if p not in new_primes]
                if not available:
                    T *= alpha
                    continue
                new_p = random.choice(available)
                new_primes.append(new_p)
                new_primes.sort()
            else:
                # Remove a prime
                if len(new_primes) <= 2:
                    T *= alpha
                    continue
                i = random.randint(0, len(new_primes) - 1)
                new_primes.pop(i)
                new_primes.sort()
        
        # Check if valid (product <= limit)
        new_n = _product_of_primes(new_primes)
        if new_n > limit or new_n < 2:
            T *= alpha
            continue
        
        # Compute score
        new_score = _tau_perp_from_primes(new_primes)
        
        # Accept or reject based on simulated annealing
        delta = new_score - current_score
        if delta > 0:
            # Always accept improvement
            current_primes = new_primes
            current_n = new_n
            current_score = new_score
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_n = current_n
                best_primes = list(current_primes)
        else:
            # Accept worse solution with probability exp(delta/T)
            if T > 0 and random.random() < math.exp(delta / T):
                current_primes = new_primes
                current_n = new_n
                current_score = new_score
        
        # Cool down
        T *= alpha
        step += 1
    
    # Final local search to polish
    if time.time() < deadline - 5:
        # Greedy improvement: try all single swaps
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_primes)):
                for p in primes:
                    if p in best_primes:
                        continue
                    new_primes = list(best_primes)
                    new_primes[i] = p
                    new_primes.sort()
                    new_n = _product_of_primes(new_primes)
                    if new_n > limit:
                        continue
                    new_score = _tau_perp_from_primes(new_primes)
                    if new_score > best_score:
                        best_score = new_score
                        best_n = new_n
                        best_primes = new_primes
                        improved = True
                        break
                if improved:
                    break
    
    return best_n
# EVOLVE-BLOCK-END
