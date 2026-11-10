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

    primes = sieve(2000)
    
    def tau_perp_fast(prime_factors):
        """Compute tau_perp for squarefree n with given distinct primes using merge-based sorted divisor generation."""
        k = len(prime_factors)
        if k == 0:
            return 0
        # Sort primes for consistent ordering
        sorted_primes = sorted(prime_factors)
        
        # Build divisors incrementally, maintaining sorted order
        # Start with [1]
        divs = [1]
        for p in sorted_primes:
            # divs is currently sorted
            # new_divs = divs + [d*p for d in divs], then merge
            new_divs = [d * p for d in divs]
            # Merge divs and new_divs (both sorted)
            merged = []
            i, j = 0, 0
            while i < len(divs) and j < len(new_divs):
                if divs[i] <= new_divs[j]:
                    merged.append(divs[i])
                    i += 1
                else:
                    merged.append(new_divs[j])
                    j += 1
            while i < len(divs):
                merged.append(divs[i])
                i += 1
            while j < len(new_divs):
                merged.append(new_divs[j])
                j += 1
            divs = merged
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Try multiple initial configurations with different numbers of primes
    best_n = None
    best_score = -1
    best_primes = None

    # Try k from 12 to 17 distinct primes
    for k in range(12, 18):
        # Start with smallest k primes
        candidate = primes[:k]
        prod = 1
        for p in candidate:
            prod *= p
        if prod <= 10**18:
            s = tau_perp_fast(candidate)
            if s > best_score:
                best_score = s
                best_n = prod
                best_primes = list(candidate)
    
    # Also try with slightly larger primes for smaller k
    for k in range(10, 13):
        # Try shifting to larger primes
        for offset in range(0, 50, 5):
            candidate = primes[offset:offset+k]
            prod = 1
            for p in candidate:
                prod *= p
            if prod <= 10**18:
                s = tau_perp_fast(candidate)
                if s > best_score:
                    best_score = s
                    best_n = prod
                    best_primes = list(candidate)

    current_primes = list(best_primes)
    current_prod = best_n
    current_score = best_score

    # Local search: swap primes
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 10.0:
        improved = False
        iteration += 1
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            p_old = current_primes[idx]
            current_prod_without = current_prod // p_old
            
            best_swap_score = -1
            best_swap_prime = -1
            
            for p_new in primes:
                if p_new * current_prod_without > 10**18:
                    break
                if p_new == p_old:
                    continue
                if p_new in current_primes:
                    continue
                
                new_primes = list(current_primes)
                new_primes[idx] = p_new
                s_new = tau_perp_fast(new_primes)
                
                if s_new > best_swap_score:
                    best_swap_score = s_new
                    best_swap_prime = p_new
            
            if best_swap_prime != -1 and best_swap_score > current_score:
                current_primes[idx] = best_swap_prime
                current_prod = current_prod_without * best_swap_prime
                current_score = best_swap_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                    best_primes = list(current_primes)
                    
            if time.time() > deadline - 5.0:
                break

    # Try adding or removing a prime
    if time.time() < deadline - 10.0:
        # Try adding a prime
        for p_new in primes:
            if time.time() > deadline - 5.0:
                break
            if p_new in current_primes:
                continue
            if current_prod * p_new > 10**18:
                break
            new_primes = list(current_primes) + [p_new]
            s_new = tau_perp_fast(new_primes)
            if s_new > best_score:
                best_score = s_new
                best_n = current_prod * p_new
                best_primes = new_primes
                current_primes = list(new_primes)
                current_prod = best_n
                current_score = s_new
                # Continue searching from new state
                improved2 = True
                while improved2 and time.time() < deadline - 10.0:
                    improved2 = False
                    indices2 = list(range(len(current_primes)))
                    random.shuffle(indices2)
                    for idx2 in indices2:
                        if time.time() > deadline - 5.0:
                            break
                        p_old2 = current_primes[idx2]
                        cpw = current_prod // p_old2
                        best_swap_score2 = -1
                        best_swap_prime2 = -1
                        for p_new2 in primes:
                            if p_new2 * cpw > 10**18:
                                break
                            if p_new2 == p_old2:
                                continue
                            if p_new2 in current_primes:
                                continue
                            new_primes2 = list(current_primes)
                            new_primes2[idx2] = p_new2
                            s_new2 = tau_perp_fast(new_primes2)
                            if s_new2 > best_swap_score2:
                                best_swap_score2 = s_new2
                                best_swap_prime2 = p_new2
                        if best_swap_prime2 != -1 and best_swap_score2 > current_score:
                            current_primes[idx2] = best_swap_prime2
                            current_prod = cpw * best_swap_prime2
                            current_score = best_swap_score2
                            improved2 = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_prod
                                best_primes = list(current_primes)
                        if time.time() > deadline - 5.0:
                            break

        # Try removing a prime
        for idx_rm in range(len(current_primes)):
            if time.time() > deadline - 5.0:
                break
            new_primes = list(current_primes)
            new_primes.pop(idx_rm)
            s_new = tau_perp_fast(new_primes)
            if s_new > best_score:
                best_score = s_new
                new_prod = 1
                for p in new_primes:
                    new_prod *= p
                best_n = new_prod
                best_primes = new_primes
                current_primes = list(new_primes)
                current_prod = new_prod
                current_score = s_new

    return best_n
