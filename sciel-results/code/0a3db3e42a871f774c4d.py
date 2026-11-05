import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def get_divisors(n):
        divs = [1]
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            if temp == 1:
                break
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
        return sorted(divs)

    def calc_score(n):
        if n < 2:
            return 0
        divs = get_divisors(n)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Helper to get factorization of n
    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors[p] = count
        if temp > 1:
            factors[temp] = 1
        return factors

    # Helper to construct n from factors
    def make_n(factors):
        n = 1
        for p, e in factors.items():
            n *= (p ** e)
        return n

    # Initial candidates from inspirations and analysis
    candidates = []
    
    # Square-free 15 primes
    n_sq15 = 1
    for p in primes[:15]:
        n_sq15 *= p
    if n_sq15 <= 10**18:
        candidates.append(n_sq15)
        
    # Inspiration 1 best: 2^5 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    n_i1 = (2**5) * (3**2)
    for p in primes[2:14]:
        n_i1 *= p
    if n_i1 <= 10**18:
        candidates.append(n_i1)
        
    # 2^4 * 3^3 * 5 * ... * 43
    n_i2 = (2**4) * (3**3)
    for p in primes[2:14]:
        n_i2 *= p
    if n_i2 <= 10**18:
        candidates.append(n_i2)
        
    # 2^6 * 3 * 5 * ... * 43
    n_i3 = (2**6) * (3**1)
    for p in primes[2:14]:
        n_i3 *= p
    if n_i3 <= 10**18:
        candidates.append(n_i3)
        
    # 2^3 * 3^4 * 5 * ... * 43
    n_i4 = (2**3) * (3**4)
    for p in primes[2:14]:
        n_i4 *= p
    if n_i4 <= 10**18:
        candidates.append(n_i4)
        
    # 2^2 * 3^5 * 5 * ... * 43
    n_i5 = (2**2) * (3**5)
    for p in primes[2:14]:
        n_i5 *= p
    if n_i5 <= 10**18:
        candidates.append(n_i5)
        
    # 2^1 * 3^6 * 5 * ... * 43
    n_i6 = (2**1) * (3**6)
    for p in primes[2:14]:
        n_i6 *= p
    if n_i6 <= 10**18:
        candidates.append(n_i6)

    # Try to maximize tau(n) with more primes
    # 2^7 * 3^3 * 5 * ... * 41
    n_i7 = (2**7) * (3**3)
    for p in primes[2:13]:
        n_i7 *= p
    if n_i7 <= 10**18:
        candidates.append(n_i7)

    # 2^8 * 3^2 * 5 * ... * 41
    n_i8 = (2**8) * (3**2)
    for p in primes[2:13]:
        n_i8 *= p
    if n_i8 <= 10**18:
        candidates.append(n_i8)

    # Evaluate initial candidates
    best_n = 0
    best_score = -1
    for n in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    if best_n == 0:
        return 2

    # Local Search
    # We will try to improve best_n by modifying its prime exponents
    # We represent n as a dict of {prime: exponent}
    
    visited = set()
    visited.add(best_n)
    
    current_n = best_n
    current_score = best_score
    
    # Limit local search steps to avoid getting stuck
    max_steps = 500
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        factors = factorize(current_n)
        improved = False
        
        # Generate neighbors
        neighbors = []
        
        # 1. Increase exponent of a prime
        for p in factors.keys():
            e = factors[p]
            new_n = current_n * p
            if new_n <= 10**18 and new_n not in visited:
                neighbors.append(new_n)
                
        # 2. Decrease exponent of a prime
        for p in factors.keys():
            e = factors[p]
            if e > 1:
                new_n = current_n // p
                if new_n not in visited:
                    neighbors.append(new_n)
            elif e == 1:
                new_n = current_n // p
                if new_n >= 2 and new_n not in visited:
                    neighbors.append(new_n)
                    
        # 3. Swap: Decrease p_i, Increase p_j
        # This is more expensive to generate, so we only do it if we have time
        if time.time() < deadline - 10:
            p_list = list(factors.keys())
            for i in range(len(p_list)):
                for j in range(len(p_list)):
                    if i == j: continue
                    p_i = p_list[i]
                    p_j = p_list[j]
                    if factors[p_i] > 0:
                        new_n = current_n // p_i * p_j
                        if new_n <= 10**18 and new_n not in visited and new_n != current_n:
                            neighbors.append(new_n)
                            
        # 4. Add a new prime factor (if not present)
        # Try small primes not in factors
        for p in primes:
            if p not in factors:
                new_n = current_n * p
                if new_n <= 10**18 and new_n not in visited:
                    neighbors.append(new_n)
                else:
                    break # Primes are sorted, so larger ones will also fail

        # Evaluate neighbors
        best_neighbor_n = current_n
        best_neighbor_score = current_score
        
        # Shuffle neighbors to avoid bias? Or just iterate.
        # Given time constraints, we might not evaluate all neighbors.
        # We can prioritize neighbors that increase tau(n) or are "smaller" in some metric.
        # For now, just evaluate up to a limit or all if fast enough.
        
        # Optimization: Sort neighbors by potential? 
        # Heuristic: Prefer neighbors with higher tau(n).
        # But calculating tau(n) is cheap.
        
        # To save time, we will only evaluate a subset of neighbors if there are too many
        # However, the number of neighbors is small (sum of exponents + number of primes + swaps)
        # Swaps can be O(k^2). k is small (~15). So ~225 swaps.
        # Total neighbors ~ 15 + 15 + 225 + 15 = ~270.
        # Evaluating 270 neighbors takes ~270 * 0.1s = 27s. Too slow for 500 steps.
        # We need to be smarter.
        
        # Strategy: Only evaluate a few promising neighbors per step.
        # Or, reduce max_steps.
        # Let's limit evaluation to the first 20 neighbors generated, or random 20.
        # Actually, let's just evaluate all if we have time, but reduce max_steps.
        
        # Let's try to evaluate all neighbors but break early if time is tight.
        for new_n in neighbors:
            if time.time() > deadline:
                break
            s = calc_score(new_n)
            visited.add(new_n)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor_n = new_n
                improved = True
                
        if not improved:
            break
            
        current_n = best_neighbor_n
        current_score = best_neighbor_score
        
        if current_score > best_score:
            best_score = current_score
            best_n = current_n

    return best_n
