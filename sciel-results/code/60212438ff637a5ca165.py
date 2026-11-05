import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    def get_divisors_fast(prime_powers):
        """Get sorted divisors from prime factorization as list of (prime, exponent) tuples."""
        divs = [1]
        for p, e in prime_powers:
            if e == 0:
                continue
            current_len = len(divs)
            power = 1
            for _ in range(e):
                power *= p
                for i in range(current_len):
                    divs.append(divs[i] * power)
        divs.sort()
        return divs
    
    def calc_score_from_divisors(divs):
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score
    
    def evaluate_factorization(powers):
        """powers: list of (prime, exponent). Returns (score, n) or (-1, -1) if invalid."""
        n = 1
        for p, e in powers:
            if e > 0:
                n *= (p ** e)
        if n > 10**18 or n < 2:
            return -1, -1
        divs = get_divisors_fast(powers)
        s = calc_score_from_divisors(divs)
        return s, n
    
    def factorize(n):
        """Return list of (prime, exponent) for n."""
        pows = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count > 0:
                pows.append((p, count))
        if temp > 1:
            pows.append((temp, 1))
        return pows
    
    best_n = 2
    best_score = 0
    
    # Precompute prefix products of primes
    prefix_prod = [1]
    for p in primes:
        prefix_prod.append(prefix_prod[-1] * p)
    
    # Find max k such that product of first k primes <= 10^18
    max_k = 0
    for i in range(1, len(primes) + 1):
        if prefix_prod[i] <= 10**18:
            max_k = i
        else:
            break
    
    # 1. Evaluate square-free products of first k primes
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        pows = [(primes[i], 1) for i in range(k)]
        s, n = evaluate_factorization(pows)
        if s > best_score and n > 0:
            best_score = s
            best_n = n
    
    # 2. Structured search with exponent variations
    for k in range(max(1, max_k - 6), max_k + 1):
        if time.time() > deadline:
            break
        base_pows = [(primes[i], 1) for i in range(k)]
        
        for i in range(min(4, k)):
            for extra in range(1, 4):
                pows = [(primes[j], 1) for j in range(k)]
                pows[i] = (primes[i], 1 + extra)
                
                n_val = 1
                for p, e in pows:
                    n_val *= (p ** e)
                
                if n_val <= 10**18:
                    s, n = evaluate_factorization(pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                else:
                    for drop_start in range(k - 1, i, -1):
                        if drop_start >= k:
                            continue
                        new_pows = pows[:drop_start]
                        if i >= drop_start:
                            continue
                        n_val = 1
                        for p, e in new_pows:
                            n_val *= (p ** e)
                        if n_val <= 10**18 and n_val >= 2:
                            s, n = evaluate_factorization(new_pows)
                            if s > best_score and n > 0:
                                best_score = s
                                best_n = n
                        break
    
    # 3. Try "almost square-free": drop one large prime, increase exponent of 2
    for k in range(max(2, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 15):
            pows = [(primes[0], a)]
            for i in range(1, k - 1):
                pows.append((primes[i], 1))
            s, n = evaluate_factorization(pows)
            if s > best_score and n > 0:
                best_score = s
                best_n = n
    
    # 4. Try dropping two large primes, increase exponents of 2 and 3
    for k in range(max(3, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 10):
            for b in range(1, 5):
                pows = [(primes[0], a), (primes[1], b)]
                for i in range(2, k - 2):
                    pows.append((primes[i], 1))
                s, n = evaluate_factorization(pows)
                if s > best_score and n > 0:
                    best_score = s
                    best_n = n
    
    # 5. Systematic search around known good region (k=15)
    if max_k >= 14:
        for k in range(max(12, max_k - 3), max_k + 1):
            if time.time() > deadline:
                break
            for a in range(1, 8):
                for b in range(1, 4):
                    for c in range(1, 3):
                        if time.time() > deadline:
                            break
                        pows = [(primes[0], a), (primes[1], b), (primes[2], c)]
                        for i in range(3, k - 1):
                            pows.append((primes[i], 1))
                        s, n = evaluate_factorization(pows)
                        if s > best_score and n > 0:
                            best_score = s
                            best_n = n
    
    # 6. Try small primes with higher exponents
    if time.time() < deadline - 20:
        for a in range(1, 61):
            if time.time() > deadline:
                break
            for b in range(1, a + 1):
                if time.time() > deadline:
                    break
                for c in range(1, b + 1):
                    if time.time() > deadline:
                        break
                    for d in range(1, c + 1):
                        if time.time() > deadline:
                            break
                        pows = [(2, a), (3, b), (5, c), (7, d)]
                        s, n = evaluate_factorization(pows)
                        if s > best_score and n > 0:
                            best_score = s
                            best_n = n
    
    # 7. Local optimization - perturb the best solution
    if best_score > 0 and time.time() < deadline - 5:
        current_pows = factorize(best_n)
        
        for _round in range(3):
            if time.time() > deadline:
                break
            improved = False
            for idx, (p, e) in enumerate(current_pows):
                if time.time() > deadline:
                    break
                if e > 1:
                    new_pows = list(current_pows)
                    new_pows[idx] = (p, e - 1)
                    s, n = evaluate_factorization(new_pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                        current_pows = factorize(best_n)
                        improved = True
                        break
                
                new_pows = list(current_pows)
                new_pows[idx] = (p, e + 1)
                s, n = evaluate_factorization(new_pows)
                if s > best_score and n > 0:
                    best_score = s
                    best_n = n
                    current_pows = factorize(best_n)
                    improved = True
                    break
            
            if not improved:
                for p in primes:
                    if time.time() > deadline:
                        break
                    if any(pp == p for pp, _ in current_pows):
                        continue
                    new_pows = current_pows + [(p, 1)]
                    s, n = evaluate_factorization(new_pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                        current_pows = factorize(best_n)
                        improved = True
                        break
            
            if not improved:
                break
    
    if best_n < 2:
        best_n = 2
        best_score = 1
    
    return best_n
