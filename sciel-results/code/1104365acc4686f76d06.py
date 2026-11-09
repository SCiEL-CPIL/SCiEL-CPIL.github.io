import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 150s limit

    best_M = 333312
    best_ks = [336, 372, 434]

    def segmented_sigma_search(max_k, chunk_size=1000000):
        """
        Search for M values with many k solutions using segmented approach.
        Returns (best_M, best_ks) found up to max_k.
        """
        m_to_k = {}
        
        def sigma(n):
            """Compute sigma(n) using trial division."""
            if n == 1:
                return 1
            result = 1
            temp = n
            d = 2
            while d * d <= temp:
                if temp % d == 0:
                    term = 1
                    power = 1
                    while temp % d == 0:
                        temp //= d
                        power *= d
                        term += power
                    result *= term
                d += 1
            if temp > 1:
                result *= (1 + temp)
            return result
        
        # For small n, use a precomputed sieve for speed
        # For larger n, use trial division
        SIEVE_LIMIT = 2000000
        sigma_small = [0] * (SIEVE_LIMIT + 1)
        sigma_small[1] = 1
        
        # Linear sieve for small values
        spf = [0] * (SIEVE_LIMIT + 1)
        prime_power = [0] * (SIEVE_LIMIT + 1)
        sigma_prime_power = [0] * (SIEVE_LIMIT + 1)
        primes = []
        
        for i in range(2, SIEVE_LIMIT + 1):
            if spf[i] == 0:
                spf[i] = i
                primes.append(i)
                prime_power[i] = i
                sigma_prime_power[i] = 1 + i
                sigma_small[i] = 1 + i
            for p in primes:
                if p > spf[i] or i * p > SIEVE_LIMIT:
                    break
                ip = i * p
                spf[ip] = p
                if p == spf[i]:
                    pp = prime_power[i] * p
                    prime_power[ip] = pp
                    ssp = sigma_prime_power[i] + pp
                    sigma_prime_power[ip] = ssp
                    sigma_small[ip] = sigma_small[i] // sigma_prime_power[i] * ssp
                    break
                else:
                    prime_power[ip] = p
                    sigma_prime_power[ip] = 1 + p
                    sigma_small[ip] = sigma_small[i] * (1 + p)
        
        for k in range(1, max_k + 1):
            if time.time() - start_time > DEADLINE:
                break
            
            if k <= SIEVE_LIMIT:
                sig = sigma_small[k]
            else:
                sig = sigma(k)
            
            M_val = k * sig
            if M_val > 10**18:
                continue
            
            if M_val not in m_to_k:
                m_to_k[M_val] = []
            m_to_k[M_val].append(k)
        
        best_M_local = 0
        best_ks_local = []
        for M_val, ks in m_to_k.items():
            valid_ks = [k for k in ks if k <= math.isqrt(M_val)]
            if len(valid_ks) > len(best_ks_local):
                best_ks_local = valid_ks
                best_M_local = M_val
        return best_M_local, best_ks_local

    # Try increasing max_k values
    max_k_targets = [1000000, 2000000, 3000000, 5000000, 8000000, 12000000, 20000000, 30000000]
    
    for max_k in max_k_targets:
        if time.time() - start_time > DEADLINE:
            break
        
        t0 = time.time()
        M, ks = segmented_sigma_search(max_k)
        
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
            if len(best_ks) >= 8:
                break
        
        elapsed = time.time() - t0
        if elapsed > 15:
            # Estimate time for next target
            if max_k >= 5000000:
                break
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
