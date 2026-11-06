import math
import time
import numpy as np
from itertools import product

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin below 150s
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def tau_perp(n, primes_list):
        """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
        # Factorize
        factors = []
        temp = n
        for p in primes_list:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        
        if not factors:
            return 0
        
        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
        
        divs.sort()
        
        if len(divs) < 2:
            return 0
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    def tau_perp_numpy(n, primes_list):
        """Faster tau_perp using numpy where possible."""
        factors = []
        temp = n
        for p in primes_list:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        
        if not factors:
            return 0
        
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
        
        divs.sort()
        
        if len(divs) < 2:
            return 0
        
        # For large divisor counts, use vectorized gcd
        if len(divs) > 1000:
            divs_arr = np.array(divs, dtype=np.int64)
            # Compute gcd of adjacent pairs using numpy
            a = divs_arr[:-1]
            b = divs_arr[1:]
            # numpy doesn't have gcd, so fall back to list comprehension
            gcds = np.array([math.gcd(int(a[i]), int(b[i])) for i in range(len(a))])
            return int(np.sum(gcds == 1))
        else:
            score = 0
            for i in range(len(divs) - 1):
                if math.gcd(divs[i], divs[i+1]) == 1:
                    score += 1
            return score
    
    # Phase 1: Generate high-quality candidates
    # Strategy: Focus on numbers with many distinct primes, low exponents
    candidates = set()
    
    def add_candidate(n):
        if 2 <= n <= max_val:
            candidates.add(n)
    
    # 1a: Square-free numbers with many primes
    sf_n = 1
    for p in primes:
        if sf_n * p <= max_val:
            sf_n *= p
        else:
            break
    add_candidate(sf_n)
    
    # Try variations: remove one prime, add exponent to another
    sf_factors = []
    temp = sf_n
    for p in primes:
        if temp % p == 0:
            sf_factors.append(p)
            temp //= p
    
    for i, p_remove in enumerate(sf_factors):
        n_var = sf_n // p_remove
        # Try adding exponent to a smaller prime
        for p_add in sf_factors:
            if p_add < p_remove and n_var * p_add <= max_val:
                add_candidate(n_var * p_add)
            # Try adding a new prime
        # Try adding a prime not in the set
        for p_new in primes:
            if p_new not in sf_factors and n_var * p_new <= max_val:
                add_candidate(n_var * p_new)
    
    # 1b: DFS with bias toward many distinct primes, low exponents
    def dfs_many_primes(idx, max_exp, current_n, current_tau, num_primes, used_primes):
        if time.time() - start_time > TIME_LIMIT - 30:
            return
        
        if current_n > 1 and num_primes >= 2:
            # Heuristic: prefer more distinct primes, lower total exponent
            total_exp = sum(used_primes.values())
            # Score: high num_primes, low total_exp
            heuristic = num_primes * 100 - total_exp * 5 + math.log(current_tau + 1)
            candidates.add(current_n)
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
        
        # Prioritize e=1 for many primes
        if num_primes >= 5:
            # Only try e=1 and e=2
            e_values = [e for e in [1, 2] if e <= e_max_possible]
        elif num_primes >= 3:
            e_values = [e for e in [1, 2, 3] if e <= e_max_possible]
        else:
            e_values = list(range(1, min(e_max_possible, 6) + 1))
        
        for e in e_values:
            if e > max_exp:
                continue
            next_n = current_n * (p ** e)
            if next_n > max_val:
                continue
            next_tau = current_tau * (e + 1)
            new_used = dict(used_primes)
            new_used[p] = new_used.get(p, 0) + e
            dfs_many_primes(idx + 1, e, next_n, next_tau, num_primes + 1, new_used)
            
            if time.time() - start_time > TIME_LIMIT - 30:
                return
    
    dfs_many_primes(0, 60, 1, 1, 0, {})
    
    # 1c: Also try the classic "high tau" DFS but with a cap on exponents
    def dfs_high_tau(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > TIME_LIMIT - 25:
            return
        
        if current_n > 1:
            candidates.add(current_n)
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
        
        # Cap exponent at 4 for this variant to encourage many primes
        e_max_capped = min(e_max_possible, 4)
        
        for e in range(1, e_max_capped + 1):
            next_n = current_n * (p ** e)
            if next_n > max_val:
                continue
            next_tau = current_tau * (e + 1)
            dfs_high_tau(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > TIME_LIMIT - 25:
                return
    
    dfs_high_tau(0, 60, 1, 1)
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    
    # Phase 2: Score candidates
    # First, do a quick pre-filter using a fast heuristic
    def quick_heuristic(n):
        """Fast estimate of tau_perp without full divisor generation."""
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        
        num_primes = len(factors)
        total_exp = sum(e for _, e in factors)
        tau = 1
        for _, e in factors:
            tau *= (e + 1)
        
        # Heuristic: coprime pairs are maximized when divisors are "spread out"
        # Rough estimate: ~ tau / (num_primes + 1) * some_factor
        # Better: use the fact that in a lattice, coprime adjacencies are rare
        # when exponents are high
        est = 0
        if num_primes == 0:
            return 0
        # Each prime contributes to "blocking" coprime pairs
        # Rough: score ~ tau * product(1/(e_i+1)) * num_primes
        block_factor = 1.0
        for _, e in factors:
            block_factor *= (1.0 / (e + 1))
        est = tau * block_factor * num_primes
        return est
    
    # Pre-filter: keep top 2000 by heuristic
    scored_candidates = []
    for n in candidates:
        h = quick_heuristic(n)
        scored_candidates.append((h, n))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [n for _, n in scored_candidates[:2000]]
    
    # Phase 3: Exact scoring
    best_score = -1
    best_n = top_candidates[0] if top_candidates else 2
    
    scored_count = 0
    for n in top_candidates:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        try:
            score = tau_perp_numpy(n, primes)
        except:
            score = tau_perp(n, primes)
        
        if score > best_score:
            best_score = score
            best_n = n
        
        scored_count += 1
    
    # Phase 4: Local search around best_n
    if time.time() - start_time < TIME_LIMIT - 10:
        # Factorize best_n
        best_factors = {}
        temp = best_n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                best_factors[p] = count
        if temp > 1:
            best_factors[temp] = 1
        
        local_candidates = set()
        
        # Try adjusting each prime's exponent by ±1
        for p in list(best_factors.keys()):
            e = best_factors[p]
            # Increase
            new_n = best_n * p
            if new_n <= max_val:
                local_candidates.add(new_n)
            # Decrease
            if e > 1:
                new_n = best_n // p
                if new_n >= 2:
                    local_candidates.add(new_n)
        
        # Try swapping a prime: remove p_i, add p_j
        for p_remove in list(best_factors.keys()):
            n_base = best_n // p_remove
            for p_add in primes:
                if p_add not in best_factors and n_base * p_add <= max_val:
                    local_candidates.add(n_base * p_add)
        
        # Try adding a new prime
        for p in primes:
            if p not in best_factors:
                new_n = best_n * p
                if new_n <= max_val:
                    local_candidates.add(new_n)
        
        # Try removing a prime
        for p in list(best_factors.keys()):
            new_n = best_n // p
            if new_n >= 2:
                local_candidates.add(new_n)
        
        # Evaluate local candidates
        for n in local_candidates:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            try:
                score = tau_perp_numpy(n, primes)
            except:
                score = tau_perp(n, primes)
            if score > best_score:
                best_score = score
                best_n = n
        
        # Second round of local search if time permits
        if time.time() - start_time < TIME_LIMIT - 5 and best_score > 0:
            # Re-factorize
            best_factors2 = {}
            temp = best_n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    count = 0
                    while temp % p == 0:
                        temp //= p
                        count += 1
                    best_factors2[p] = count
            if temp > 1:
                best_factors2[temp] = 1
            
            local_candidates2 = set()
            
            for p in list(best_factors2.keys()):
                e = best_factors2[p]
                new_n = best_n * p
                if new_n <= max_val:
                    local_candidates2.add(new_n)
                if e > 1:
                    new_n = best_n // p
                    if new_n >= 2:
                        local_candidates2.add(new_n)
            
            for p_remove in list(best_factors2.keys()):
                n_base = best_n // p_remove
                for p_add in primes:
                    if p_add not in best_factors2 and n_base * p_add <= max_val:
                        local_candidates2.add(n_base * p_add)
            
            for p in primes:
                if p not in best_factors2:
                    new_n = best_n * p
                    if new_n <= max_val:
                        local_candidates2.add(new_n)
            
            for p in list(best_factors2.keys()):
                new_n = best_n // p
                if new_n >= 2:
                    local_candidates2.add(new_n)
            
            for n in local_candidates2:
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
                try:
                    score = tau_perp_numpy(n, primes)
                except:
                    score = tau_perp(n, primes)
                if score > best_score:
                    best_score = score
                    best_n = n
    
    return best_n
