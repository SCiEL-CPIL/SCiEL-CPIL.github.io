"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _sieve(limit):
    """Generate primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _tau_perp_from_primes(prime_list):
    """Compute tau_perp for a squarefree number given its sorted prime list.
    Generates all divisors, sorts them, counts adjacent coprime pairs."""
    k = len(prime_list)
    if k == 0:
        return 0
    # Generate all 2^k divisors using bit manipulation
    divisors = [1]
    for p in prime_list:
        current_len = len(divisors)
        for i in range(current_len):
            divisors.append(divisors[i] * p)
    divisors.sort()
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def _tau_perp_fast(n, primes):
    """Fast tau_perp computation: factor n, generate divisors, count coprime adjacents."""
    temp = n
    factors = []
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            factors.append((p, cnt))
    if temp > 1:
        factors.append((temp, 1))
    
    # Generate all divisors
    divisors = [1]
    for p, e in factors:
        current_len = len(divisors)
        mult = 1
        for _ in range(e):
            mult *= p
            for i in range(current_len):
                divisors.append(divisors[i] * mult)
    
    divisors.sort()
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    limit = 10**18
    
    # Generate primes up to a reasonable limit
    prime_limit = 20000
    primes = _sieve(prime_limit)
    
    # Initial solution: product of smallest primes (squarefree)
    base_primes = []
    n = 1
    for p in primes:
        if n * p > limit:
            break
        base_primes.append(p)
        n *= p
    
    best_n = n
    best_score = _tau_perp_from_primes(base_primes)
    
    # Greedy improvement: try swapping each prime with a larger one
    improved = True
    while improved:
        improved = False
        for i in range(len(base_primes)):
            if time.time() > deadline - 10:
                return best_n
            for j in range(len(base_primes) - 1, -1, -1):
                if j <= i:
                    break
                # Try replacing base_primes[i] with primes[j]
                new_p = primes[j]
                if new_p in base_primes:
                    continue
                candidate_product = n // base_primes[i] * new_p
                if candidate_product > limit:
                    continue
                new_list = base_primes[:]
                new_list[i] = new_p
                new_list.sort()
                score = _tau_perp_from_primes(new_list)
                if score > best_score:
                    best_score = score
                    best_n = candidate_product
                    base_primes = new_list
                    n = candidate_product
                    improved = True
                    break
    
    # Simulated annealing
    if time.time() < deadline - 30:
        current_n = best_n
        current_list = base_primes[:]
        current_score = best_score
        temperature = 10.0
        cooling = 0.9995
        
        while time.time() < deadline - 10:
            # Pick a random prime to swap out
            idx = random.randint(0, len(current_list) - 1)
            old_p = current_list[idx]
            
            # Pick a random new prime (not already in list)
            new_p = primes[random.randint(0, len(primes) - 1)]
            if new_p in current_list:
                continue
            
            candidate_n = current_n // old_p * new_p
            if candidate_n > limit:
                continue
            
            new_list = current_list[:]
            new_list[idx] = new_p
            new_list.sort()
            new_score = _tau_perp_from_primes(new_list)
            
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_n = candidate_n
                current_list = new_list
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
            
            temperature *= cooling
            if temperature < 0.1:
                temperature = 0.1
    
    # Final local search polish
    if time.time() < deadline - 5:
        # Extract current prime list from best_n
        temp = best_n
        final_primes = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt == 1:
                    final_primes.append(p)
                else:
                    # Not squarefree, handle differently
                    pass
        if temp > 1:
            final_primes.append(temp)
        
        # Try single swaps again with full prime pool
        for i in range(len(final_primes)):
            if time.time() > deadline - 2:
                break
            old_p = final_primes[i]
            for p in primes:
                if p == old_p or p in final_primes:
                    continue
                candidate_n = best_n // old_p * p
                if candidate_n > limit:
                    break
                new_list = final_primes[:]
                new_list[i] = p
                new_list.sort()
                score = _tau_perp_from_primes(new_list)
                if score > best_score:
                    best_score = score
                    best_n = candidate_n
                    final_primes = new_list
    
    return best_n
# EVOLVE-BLOCK-END
