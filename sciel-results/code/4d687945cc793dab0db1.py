import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Generate primes up to a larger limit to allow more search space
    primes = sieve(5000)
    
    def tau_perp_fast(n, prime_factors):
        """Compute tau_perp using prime support sets for faster gcd checks."""
        # Generate all divisors with their prime support masks
        # For square-free n, each divisor corresponds to a subset of primes
        k = len(prime_factors)
        divs = [(1, 0)]  # (value, bitmask)
        
        for i, p in enumerate(prime_factors):
            new_divs = []
            for val, mask in divs:
                new_divs.append((val * p, mask | (1 << i)))
            divs.extend(new_divs)
        
        divs.sort(key=lambda x: x[0])
        
        score = 0
        for i in range(len(divs) - 1):
            # Two divisors are coprime iff their prime supports are disjoint
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    # Initialize with smallest primes, but use greedy set-packing heuristic
    # to select primes that maximize coprime adjacency
    
    def greedy_prime_selection(prime_list, max_product=10**18, target_k=15):
        """Select primes using greedy set-packing to maximize coprime adjacency."""
        selected = []
        prod = 1
        
        # Start with smallest primes for base structure
        for p in prime_list:
            if len(selected) >= target_k:
                break
            if prod * p > max_product:
                break
            selected.append(p)
            prod *= p
        
        return selected, prod

    # Initial selection
    initial_primes, prod = greedy_prime_selection(primes)
    
    best_n = prod
    best_score = tau_perp_fast(prod, initial_primes)
    best_primes = list(initial_primes)

    # Local Search with improved candidate evaluation
    current_primes = list(initial_primes)
    current_prod = prod
    current_score = best_score
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 15.0:
        improved = False
        iteration += 1
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10.0:
                break
            p_old = current_primes[idx]
            
            max_p_new = int((10**18 * p_old) / current_prod)
            
            best_swap_score = -1
            best_swap_prime = -1
            best_swap_prod = -1
            
            current_set = set(current_primes)
            
            # Sample candidates rather than exhaustive search for speed
            # Focus on primes that could improve the set-packing structure
            candidates = []
            for p_new in primes:
                if p_new > max_p_new:
                    break
                if p_new == p_old:
                    continue
                if p_new in current_set:
                    continue
                candidates.append(p_new)
            
            # If too many candidates, sample a subset
            if len(candidates) > 200:
                random.shuffle(candidates)
                candidates = candidates[:200]
            
            for p_new in candidates:
                new_prod = (current_prod // p_old) * p_new
                new_primes = list(current_primes)
                new_primes[idx] = p_new
                s_new = tau_perp_fast(new_prod, new_primes)
                
                if s_new > best_swap_score:
                    best_swap_score = s_new
                    best_swap_prime = p_new
                    best_swap_prod = new_prod
            
            if best_swap_prime != -1 and best_swap_score > current_score:
                current_primes[idx] = best_swap_prime
                current_prod = best_swap_prod
                current_score = best_swap_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                    best_primes = list(current_primes)
                    
            if time.time() > deadline - 10.0:
                break

    # Try to add more primes if possible (increase k)
    if time.time() < deadline - 10.0:
        for p in primes:
            if p in best_primes:
                continue
            if best_n * p > 10**18:
                break
            new_primes = best_primes + [p]
            new_n = best_n * p
            s_new = tau_perp_fast(new_n, new_primes)
            if s_new > best_score:
                best_score = s_new
                best_n = new_n
                best_primes = new_primes

    # Try to remove a prime and add two smaller ones if beneficial
    if time.time() < deadline - 10.0:
        for i in range(len(best_primes)):
            p_remove = best_primes[i]
            base_prod = best_n // p_remove
            base_primes = best_primes[:i] + best_primes[i+1:]
            
            # Try adding two primes
            for p1 in primes:
                if p1 in base_primes:
                    continue
                if base_prod * p1 > 10**18:
                    break
                for p2 in primes:
                    if p2 <= p1:
                        continue
                    if p2 in base_primes or p2 == p1:
                        continue
                    if base_prod * p1 * p2 > 10**18:
                        break
                    new_primes = sorted(base_primes + [p1, p2])
                    new_n = base_prod * p1 * p2
                    s_new = tau_perp_fast(new_n, new_primes)
                    if s_new > best_score:
                        best_score = s_new
                        best_n = new_n
                        best_primes = new_primes
                        break
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break

    return best_n
