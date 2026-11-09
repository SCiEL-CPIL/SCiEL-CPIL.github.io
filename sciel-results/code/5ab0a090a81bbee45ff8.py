"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    limit = 10**18
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    base_primes = primes[:15]
    n = 1
    for p in base_primes:
        n *= p
    
    def compute_tau_perp(n):
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.append(p)
                temp //= p
        if temp > 1:
            factors.append(temp)
        
        divisors = [1]
        for p in factors:
            new_divs = []
            for d in divisors:
                new_divs.append(d * p)
            divisors.extend(new_divs)
        
        divisors.sort()
        
        count = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                count += 1
        return count
    
    best_n = n
    best_score = compute_tau_perp(n)
    
    # Single swaps
    base_product = n
    for i in range(len(base_primes)):
        if time.time() > deadline:
            break
        for j in range(len(base_primes), len(primes)):
            if time.time() > deadline:
                break
            candidate = base_product // base_primes[i] * primes[j]
            if candidate > limit:
                break
            if candidate == n:
                continue
            score = compute_tau_perp(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
                base_product = candidate
                base_primes[i] = primes[j]
                base_primes.sort()
    
    # Double swaps
    if time.time() < deadline - 20:
        base_product = best_n
        # Re-extract primes from best_n
        temp = best_n
        current_primes = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                current_primes.append(p)
                temp //= p
        if temp > 1:
            current_primes.append(temp)
        
        for i in range(len(current_primes)):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, len(current_primes)):
                if time.time() > deadline - 5:
                    break
                for a in range(len(current_primes), len(primes)):
                    if time.time() > deadline - 5:
                        break
                    for b in range(a+1, len(primes)):
                        if time.time() > deadline - 5:
                            break
                        candidate = base_product // current_primes[i] // current_primes[j] * primes[a] * primes[b]
                        if candidate > limit:
                            break
                        if candidate == best_n:
                            continue
                        score = compute_tau_perp(candidate)
                        if score > best_score:
                            best_score = score
                            best_n = candidate
                            base_product = candidate
                            # Update current_primes
                            current_primes = list(current_primes)
                            current_primes[i] = primes[a]
                            current_primes[j] = primes[b]
                            current_primes.sort()
    
    return best_n
# EVOLVE-BLOCK-END
