import time
import math
from math import gcd

_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _elapsed(t0):
    return time.time() - t0

def _time_exhausted(t0, margin=2.0):
    return _elapsed(t0) > (_DEADLINE - margin)

def _score(chosen):
    """Compute sum 1/(512 - a) as float."""
    s = 0.0
    for a in chosen:
        s += 1.0 / (512.0 - a)
    return s

def _get_prime_factors(n, primes):
    """Get unique prime factors of n."""
    factors = set()
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.add(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.add(temp)
    return factors

def construct():
    t0 = time.time()
    
    # Sieve of Eratosthenes for primes up to 512
    limit = 512
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Pre-compute prime factors for all numbers 1..511
    prime_factors = [set() for _ in range(limit)]
    for i in range(1, limit):
        prime_factors[i] = _get_prime_factors(i, primes)
        
    # Greedy Initialization
    # We want to pick numbers with large 1/(512-a), i.e., large a.
    # Check if a is coprime to all chosen.
    # This is equivalent to: prime_factors[a] is disjoint from union of prime_factors[chosen].
    
    chosen = []
    chosen_prime_factors = set()
    
    for a in range(511, 0, -1):
        if not prime_factors[a]: # 1 has no prime factors, gcd(1, x) = 1 always
            chosen.append(a)
            # 1 doesn't add any prime factors to the forbidden set
        else:
            # Check disjointness
            if not prime_factors[a].intersection(chosen_prime_factors):
                chosen.append(a)
                chosen_prime_factors.update(prime_factors[a])
                
    best_score = _score(chosen)
    
    # Local Search with multiple move types
    # 1. Swap: remove one element, add one better element
    # 2. Add: try to add an element that fits (increases count)
    # 3. Remove+Add2: remove one, add two (if beneficial)
    
    improved = True
    while improved and not _time_exhausted(t0):
        improved = False
        chosen_set = set(chosen)
        chosen_factors_union = set()
        for a in chosen:
            chosen_factors_union.update(prime_factors[a])
        
        # Try swaps
        for i in range(len(chosen)):
            if _time_exhausted(t0):
                break
            
            current = chosen[i]
            current_score = 1.0 / (512.0 - current)
            
            # Determine required prime factors (from all other chosen elements)
            req_factors = set()
            for j in range(len(chosen)):
                if j != i:
                    req_factors.update(prime_factors[chosen[j]])
            
            # Find the best replacement
            best_new = -1
            best_new_score = 0.0
            
            # Iterate candidates in descending order of score (descending a)
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if a == current:
                    continue
                
                # Check if a is coprime to all other chosen
                if prime_factors[a] and prime_factors[a].intersection(req_factors):
                    continue
                
                # Check if a is better
                new_score = 1.0 / (512.0 - a)
                if new_score > current_score:
                    best_new = a
                    best_new_score = new_score
                    break # Found the best candidate for this slot
            
            if best_new != -1:
                # Perform swap
                chosen[i] = best_new
                chosen_set.remove(current)
                chosen_set.add(best_new)
                best_score = best_score - current_score + best_new_score
                improved = True
                break
    
    # Try to add elements (increase set size)
    if not _time_exhausted(t0):
        added = True
        while added and not _time_exhausted(t0):
            added = False
            chosen_set = set(chosen)
            chosen_factors_union = set()
            for a in chosen:
                chosen_factors_union.update(prime_factors[a])
            
            # Try adding the best available element
            best_add = -1
            best_add_score = 0.0
            
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                
                # Check if a is coprime to all chosen
                if prime_factors[a] and prime_factors[a].intersection(chosen_factors_union):
                    continue
                
                # Check if a is worth adding (positive contribution)
                add_score = 1.0 / (512.0 - a)
                if add_score > 0:
                    best_add = a
                    best_add_score = add_score
                    break
            
            if best_add != -1:
                chosen.append(best_add)
                best_score += best_add_score
                added = True
    
    # Try remove+add2 moves: remove one element, add two better ones
    if not _time_exhausted(t0):
        improved2 = True
        while improved2 and not _time_exhausted(t0):
            improved2 = False
            chosen_set = set(chosen)
            
            for i in range(len(chosen)):
                if _time_exhausted(t0):
                    break
                
                current = chosen[i]
                current_score = 1.0 / (512.0 - current)
                
                # Determine required prime factors (from all other chosen elements)
                req_factors = set()
                for j in range(len(chosen)):
                    if j != i:
                        req_factors.update(prime_factors[chosen[j]])
                
                # Try to find two elements that together are better than current
                # and are coprime to each other and to req_factors
                best_pair = None
                best_pair_score = 0.0
                
                # Iterate pairs in order of potential score
                # We want the two largest a values that fit
                for a1 in range(511, 0, -1):
                    if _time_exhausted(t0):
                        break
                    if a1 in chosen_set or a1 == current:
                        continue
                    if prime_factors[a1] and prime_factors[a1].intersection(req_factors):
                        continue
                    
                    score1 = 1.0 / (512.0 - a1)
                    
                    for a2 in range(511, 0, -1):
                        if a2 <= a1:
                            break
                        if a2 in chosen_set or a2 == current:
                            continue
                        if prime_factors[a2] and prime_factors[a2].intersection(req_factors):
                            continue
                        
                        # Check if a1 and a2 are coprime
                        if prime_factors[a1] and prime_factors[a1].intersection(prime_factors[a2]):
                            continue
                        
                        score2 = 1.0 / (512.0 - a2)
                        pair_score = score1 + score2
                        
                        if pair_score > current_score and pair_score > best_pair_score:
                            best_pair = (a1, a2)
                            best_pair_score = pair_score
                        
                        # Early termination: if we found a pair that's clearly better,
                        # and we're iterating in descending order, we might stop early
                        # But we need to be careful about the ordering
                        if best_pair_score > current_score and a1 < 400:
                            break
                    
                    if best_pair and best_pair_score > current_score and a1 < 300:
                        break
                
                if best_pair and best_pair_score > current_score:
                    # Perform the move
                    chosen.pop(i)
                    chosen_set.remove(current)
                    chosen.append(best_pair[0])
                    chosen.append(best_pair[1])
                    chosen_set.add(best_pair[0])
                    chosen_set.add(best_pair[1])
                    best_score = best_score - current_score + best_pair_score
                    improved2 = True
                    break
    
    return chosen
