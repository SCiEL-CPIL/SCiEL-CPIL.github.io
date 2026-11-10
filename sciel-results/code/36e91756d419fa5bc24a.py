import time
import math
import random
from itertools import combinations

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

    primes = sieve(20000)
    
    def tau_perp(n):
        temp = n
        factors = []
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.append(d)
                while temp % d == 0:
                    temp //= d
            else:
                d += 1
        if temp > 1:
            factors.append(temp)
            
        divs = [1]
        for p in factors:
            new_divs = []
            for d in divs:
                new_divs.append(d * p)
            divs.extend(new_divs)
            
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Greedy Build-up Strategy
    # 1. Determine max k
    prod = 1
    max_k = 0
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    best_n = 0
    best_score = 0
    
    # We will try multiple random restarts of the greedy build-up
    # to escape local optima of the greedy choice.
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
            
        current_set = []
        current_n = 1
        current_score = 0
        
        # Greedily add primes
        for _ in range(max_k):
            best_candidate = None
            best_candidate_score = -1
            best_candidate_n = -1
            
            # Filter candidates: must not be in current_set, product must fit
            # We don't check all primes every time, just a subset to save time?
            # No, checking all is fine for 20000 primes if we break early on product.
            # But we need to evaluate score for each. That's expensive.
            # Optimization: Only evaluate a random subset of candidates?
            # Or just the first few that fit?
            
            # Let's try a random subset of 50 candidates that fit.
            candidates = []
            for p in primes:
                if p in current_set:
                    continue
                if current_n * p > 10**18:
                    break
                candidates.append(p)
                
            if not candidates:
                break
                
            # Randomly pick a subset to evaluate to speed up
            eval_set = random.sample(candidates, min(50, len(candidates)))
            
            for p in eval_set:
                new_n = current_n * p
                # Score calculation is the bottleneck
                # We can approximate or just compute
                s = tau_perp(new_n)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = p
                    best_candidate_n = new_n
            
            if best_candidate is None:
                break
                
            current_set.append(best_candidate)
            current_n = best_candidate_n
            current_score = best_candidate_score
            
        # After greedy build-up, do local search
        # Swap primes to improve score
        current_set_list = list(current_set)
        current_n_val = current_n
        current_score_val = current_score
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            indices = list(range(len(current_set_list)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                p_old = current_set_list[idx]
                max_p_new = (10**18 * p_old) // current_n_val
                
                # Find candidate to swap
                # Look for primes not in set, <= max_p_new
                # To make it faster, only check a random subset of valid primes
                valid_candidates = []
                for p in primes:
                    if p in current_set_list:
                        continue
                    if p > max_p_new:
                        break
                    valid_candidates.append(p)
                    
                if not valid_candidates:
                    continue
                
                eval_set = random.sample(valid_candidates, min(20, len(valid_candidates)))
                
                best_swap = None
                best_swap_score = -1
                best_swap_n = -1
                
                for p_new in eval_set:
                    new_n = (current_n_val // p_old) * p_new
                    s = tau_perp(new_n)
                    if s > best_swap_score:
                        best_swap_score = s
                        best_swap = p_new
                        best_swap_n = new_n
                        
                if best_swap is not None and best_swap_score > current_score_val:
                    current_score_val = best_swap_score
                    current_n_val = best_swap_n
                    current_set_list[idx] = best_swap
                    improved = True
                    break # Restart inner loop
                    
        if current_score_val > best_score:
            best_score = current_score_val
            best_n = current_n_val
            
    return best_n if best_n > 0 else 2
