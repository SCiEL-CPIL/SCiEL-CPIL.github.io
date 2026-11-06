"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Sieve for smallest prime factor up to 20000
    MAX_SIEVE = 20000
    spf = list(range(MAX_SIEVE + 1))
    for i in range(2, int(MAX_SIEVE**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SIEVE + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def prime_factors(n):
        """Return set of prime factors of n using SPF table."""
        factors = set()
        while n > 1 and n <= MAX_SIEVE:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        if n > 1:
            factors.add(n)
        return factors
    
    def count_primes(nums):
        """Count distinct primes dividing all pairwise sums of distinct elements."""
        primes = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                if s <= MAX_SIEVE:
                    while s > 1:
                        p = spf[s]
                        primes.add(p)
                        while s % p == 0:
                            s //= p
                else:
                    primes |= prime_factors(s)
        return len(primes), primes
    
    def evaluate_ks(ks, M):
        """Evaluate total distinct primes for nums = [k*M for k in ks]."""
        # Start with prime factors of M
        primes = set()
        m = M
        while m > 1 and m <= MAX_SIEVE:
            p = spf[m]
            primes.add(p)
            while m % p == 0:
                m //= p
        if m > 1:
            primes.add(m)
        
        n = len(ks)
        for i in range(n):
            for j in range(i+1, n):
                s = ks[i] + ks[j]
                if s <= MAX_SIEVE:
                    while s > 1:
                        p = spf[s]
                        primes.add(p)
                        while s % p == 0:
                            s //= p
                else:
                    primes |= prime_factors(s)
        return len(primes), primes
    
    best_count = float('inf')
    best_nums = None
    
    # Precompute smooth M candidates (few prime factors)
    # M = 2^a * 3^b * 5^c * 7^d * 11^e ... with small omega
    smooth_Ms = set()
    for a in range(0, 14):  # 2^a up to 16384
        v = (1 << a)
        if v > 10000:
            break
        for b in range(0, 10):  # 3^b up to 59049
            v2 = v * (3**b)
            if v2 > 10000:
                break
            for c in range(0, 7):  # 5^c up to 78125
                v3 = v2 * (5**c)
                if v3 > 10000:
                    break
                for d in range(0, 6):  # 7^d up to 117649
                    v4 = v3 * (7**d)
                    if v4 > 10000:
                        break
                    smooth_Ms.add(v4)
                    # Also try multiplying by 11, 13
                    for extra in [11, 13, 17]:
                        if v4 * extra <= 10000:
                            smooth_Ms.add(v4 * extra)
    
    # Filter: need at least 12 distinct positive integers k with k*M <= 10000
    # i.e., 10000 // M >= 12
    valid_Ms = [M for M in smooth_Ms if 10000 // M >= 12]
    
    # Sort by number of prime factors (fewer is better)
    def omega(m):
        """Count distinct prime factors of m."""
        count = 0
        while m > 1 and m <= MAX_SIEVE:
            p = spf[m]
            count += 1
            while m % p == 0:
                m //= p
        if m > 1:
            count += 1
        return count
    
    valid_Ms.sort(key=lambda m: (omega(m), m))
    
    # Also add some non-smooth but useful M values
    for M in [6, 10, 12, 14, 15, 18, 20, 21, 22, 24, 26, 28, 30, 33, 34, 35, 36, 38, 39, 40, 42, 44, 45, 46, 48, 50]:
        if 10000 // M >= 12:
            if M not in valid_Ms:
                valid_Ms.append(M)
    valid_Ms.sort(key=lambda m: (omega(m), m))
    
    # Local search function for optimizing ks given M
    def local_search_ks(ks, M, max_iters=2000):
        """Improve ks to minimize distinct primes of M*(k_i + k_j)."""
        n = len(ks)
        K_max = 10000 // M
        ks = sorted(ks)
        current_score, _ = evaluate_ks(ks, M)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            improved = False
            
            # Try replacing one k with another value
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old_k = ks[i]
                # Try several candidate replacements
                candidates = []
                # Nearby values
                for dk in range(-5, 6):
                    nk = old_k + dk
                    if 1 <= nk <= K_max and nk != old_k and nk not in ks:
                        candidates.append(nk)
                # Random candidates
                for _ in range(5):
                    nk = random.randint(1, K_max)
                    if nk != old_k and nk not in ks:
                        candidates.append(nk)
                
                for nk in candidates:
                    ks[i] = nk
                    new_score, _ = evaluate_ks(ks, M)
                    if new_score < current_score:
                        current_score = new_score
                        ks = sorted(ks)
                        improved = True
                        break
                    ks[i] = old_k
                if improved:
                    break
            
            if not improved:
                # Try swapping two values
                for i in range(n):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i+1, n):
                        if time.time() > deadline - 2:
                            break
                        ki, kj = ks[i], ks[j]
                        # Try new values
                        for nk_i in [ki-1, ki+1, ki-2, ki+2]:
                            for nk_j in [kj-1, kj+1, kj-2, kj+2]:
                                if 1 <= nk_i <= K_max and 1 <= nk_j <= K_max:
                                    if nk_i == nk_j:
                                        continue
                                    if nk_i in ks or nk_j in ks:
                                        continue
                                    if nk_i == ki and nk_j == kj:
                                        continue
                                    ks[i], ks[j] = nk_i, nk_j
                                    new_score, _ = evaluate_ks(ks, M)
                                    if new_score < current_score:
                                        current_score = new_score
                                        ks = sorted(ks)
                                        improved = True
                                        break
                                    ks[i], ks[j] = ki, kj
                            if improved:
                                break
                    if improved:
                        break
            if not improved:
                break
        
        return ks, current_score
    
    # Main search over M values
    for M in valid_Ms[:50]:  # Limit to top 50 smooth M's
        if time.time() > deadline - 5:
            break
        
        K_max = 10000 // M
        if K_max < 12:
            continue
        
        # Initial ks: 1, 2, ..., 12
        ks_init = list(range(1, 13))
        
        # Evaluate initial
        score_init, _ = evaluate_ks(ks_init, M)
        
        # Local search
        ks_opt, score_opt = local_search_ks(ks_init[:], M, max_iters=500)
        
        if score_opt < best_count:
            best_count = score_opt
            best_nums = [k * M for k in ks_opt]
        
        # Try a few different initial configurations
        if time.time() < deadline - 10:
            # Try spread out
            if K_max >= 24:
                ks_alt = list(range(1, 13)) * [2]  # 1,2,...,12,1,2,...,12 - no, need distinct
                # Try: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 (odds)
                if K_max >= 23:
                    ks_odds = list(range(1, 24, 2))[:12]
                    ks_odds_opt, score_odds = local_search_ks(ks_odds[:], M, max_iters=300)
                    if score_odds < best_count:
                        best_count = score_odds
                        best_nums = [k * M for k in ks_odds_opt]
            
            # Try: 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34
            if K_max >= 34:
                ks_step3 = [1 + 3*i for i in range(12)]
                ks_step3_opt, score_step3 = local_search_ks(ks_step3[:], M, max_iters=300)
                if score_step3 < best_count:
                    best_count = score_step3
                    best_nums = [k * M for k in ks_step3_opt]
    
    # Also try direct random search as backup
    if time.time() < deadline - 5:
        for attempt in range(100):
            if time.time() > deadline - 2:
                break
            nums = random.sample(range(1, 10001), 12)
            c, _ = count_primes(nums)
            if c < best_count:
                best_count = c
                best_nums = nums[:]
    
    if best_nums is None:
        return list(range(1, 13))
    
    # Verify distinct and in range
    assert len(set(best_nums)) == 12
    assert all(1 <= x <= 10000 for x in best_nums)
    
    return best_nums
