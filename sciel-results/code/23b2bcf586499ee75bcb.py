import time
import math
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n+1) if is_prime[i]]

    primes = sieve(200000)
    prime_set_global = set(primes)

    def tau_perp_numpy(prime_factors):
        k = len(prime_factors)
        if k == 0:
            return 0
        
        # Generate all divisors as (value, mask)
        # Use iterative doubling
        divs = [(1, 0)]
        for i, p in enumerate(prime_factors):
            bit = 1 << i
            current_len = len(divs)
            for j in range(current_len):
                val, mask = divs[j]
                divs.append((val * p, mask | bit))
        
        # Sort by value
        divs.sort(key=lambda x: x[0])
        
        # Count coprime adjacent pairs
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    def tau_perp_fast(prime_factors):
        """Optimized version using list operations"""
        k = len(prime_factors)
        if k == 0:
            return 0
        
        # Build divisors
        divs = [(1, 0)]
        for i, p in enumerate(prime_factors):
            bit = 1 << i
            current_len = len(divs)
            # Use list comprehension for speed
            new_divs = [(divs[j][0] * p, divs[j][1] | bit) for j in range(current_len)]
            divs.extend(new_divs)
        
        # Sort by value
        divs.sort(key=lambda x: x[0])
        
        # Count coprime adjacent pairs - vectorized approach
        masks = np.array([d[1] for d in divs], dtype=np.int64)
        # Adjacent pairs are coprime if AND is 0
        adjacent_and = masks[:-1] & masks[1:]
        score = np.sum(adjacent_and == 0)
        return int(score)

    # Initialize with smallest 15 primes
    max_k = 15
    current_primes = list(primes[:max_k])
    current_prod = 1
    for p in current_primes:
        current_prod *= p
        
    best_n = current_prod
    best_score = tau_perp_fast(current_primes)
    current_score = best_score

    def check_constraint(p_new, p_old, prod, limit=10**18):
        return p_new * prod <= limit * p_old

    # Local search
    improved = True
    no_improvement_count = 0
    max_no_improvement = 8
    iteration = 0

    while improved and time.time() < deadline - 15.0:
        improved = False
        iteration += 1
        current_prime_set = set(current_primes)
        
        # Try single prime swaps - focus on most promising candidates
        # Sort positions by prime value (larger primes are more likely to be replaceable)
        indices = sorted(range(len(current_primes)), key=lambda i: current_primes[i], reverse=True)
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10.0:
                return best_n
                
            p_old = current_primes[idx]
            
            # Find valid replacement primes
            # p_new * current_prod <= 10^18 * p_old
            # p_new <= 10^18 * p_old / current_prod
            max_p_new = (10**18 * p_old) // current_prod
            
            # Candidates: primes not in current set, <= max_p_new
            # Prefer small primes for density, but also try some larger ones
            candidates = []
            for p_new in primes:
                if p_new in current_prime_set:
                    continue
                if p_new > max_p_new:
                    break
                candidates.append(p_new)
            
            if not candidates:
                continue
            
            # Try a strategic subset: smallest 30, some middle, largest 10
            test_candidates = []
            if len(candidates) > 50:
                test_candidates.extend(candidates[:30])
                mid_start = len(candidates) // 2
                test_candidates.extend(candidates[mid_start:mid_start+10])
                test_candidates.extend(candidates[-10:])
            else:
                test_candidates = candidates
            
            best_candidate = None
            best_candidate_score = current_score
            
            for p_new in test_candidates:
                if p_new in current_prime_set:
                    continue
                if p_new * current_prod > 10**18 * p_old:
                    continue
                
                new_primes = current_primes.copy()
                new_primes[idx] = p_new
                
                s_new = tau_perp_fast(new_primes)
                
                if s_new > best_candidate_score:
                    best_candidate_score = s_new
                    best_candidate = p_new
            
            if best_candidate is not None and best_candidate_score > current_score:
                current_primes[idx] = best_candidate
                current_prod = (current_prod // p_old) * best_candidate
                current_score = best_candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                improved = True
                break
        
        # If no improvement, try random restarts from perturbed states
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                if time.time() < deadline - 25.0:
                    # Random restart: pick random 15 primes that fit
                    attempts = 0
                    while attempts < 50 and time.time() < deadline - 15.0:
                        attempts += 1
                        # Start with small primes, randomly replace some
                        perturbed = list(primes[:max_k])
                        for _ in range(random.randint(1, 4)):
                            idx = random.randint(0, max_k - 1)
                            prod_without = 1
                            for j, p in enumerate(perturbed):
                                if j != idx:
                                    prod_without *= p
                            max_p = (10**18) // prod_without
                            valid = [p for p in primes if p not in set(perturbed) and p <= max_p]
                            if valid:
                                perturbed[idx] = random.choice(valid)
                        
                        prod = 1
                        for p in perturbed:
                            prod *= p
                        if prod <= 10**18:
                            s = tau_perp_fast(perturbed)
                            if s > best_score:
                                best_score = s
                                best_n = prod
                                current_primes = perturbed
                                current_score = s
                                current_prod = prod
                                improved = True
                                no_improvement_count = 0
                                break
                
                if not improved:
                    break
            else:
                # Small perturbation: swap one random prime
                if time.time() < deadline - 20.0:
                    idx = random.randint(0, len(current_primes) - 1)
                    p_old = current_primes[idx]
                    max_p_new = (10**18 * p_old) // current_prod
                    valid = [p for p in primes if p not in set(current_primes) and p <= max_p_new]
                    if valid:
                        p_new = random.choice(valid)
                        new_primes = current_primes.copy()
                        new_primes[idx] = p_new
                        s_new = tau_perp_fast(new_primes)
                        if s_new >= current_score:  # Accept equal or better to escape plateaus
                            current_primes = new_primes
                            current_prod = (current_prod // p_old) * p_new
                            current_score = s_new
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_prod
                            no_improvement_count = 0

    return best_n
