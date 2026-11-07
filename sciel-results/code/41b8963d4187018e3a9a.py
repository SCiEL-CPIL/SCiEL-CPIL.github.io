import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    def time_check():
        return time.time() > deadline
    
    # Generate primes up to a reasonable limit
    # For prime powers p^a where p^a <= 10^18, p can be up to 10^18
    # But for k*sigma(k) <= 10^18, we need p^a * sigma(p^a) <= 10^18
    # sigma(p^a) ~ p^a for large p, so p^(2a) <= 10^18 roughly
    # For a=1: p^2 * (p+1) <= 10^18, so p <= ~10^6
    # For a=2: p^4 * (p^2+p+1) <= 10^18, p <= ~10^4
    # etc.
    
    # Let's generate primes up to 200000 for single prime powers
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(500000)
    
    # Map M -> list of k
    m_to_ks = defaultdict(list)
    prime_powers = []  # list of (k_val, sigma_val, p)
    
    for p in primes:
        if time_check():
            break
        pk = p
        sigma = 1 + p
        while pk * sigma <= 10**18:
            prime_powers.append((pk, sigma, p))
            if pk not in m_to_ks[pk * sigma]:
                m_to_ks[pk * sigma].append(pk)
            pk *= p
            sigma += pk
            if pk > 10**18:
                break
    
    # Combine pairs of prime powers with different primes
    n_pp = len(prime_powers)
    
    # Sort by k value for efficient pair search
    prime_powers.sort(key=lambda x: x[0])
    
    # Use more prime powers for pairing
    max_pp_for_pairs = min(n_pp, 20000)
    pp_for_pairs = prime_powers[:max_pp_for_pairs]
    
    for i in range(len(pp_for_pairs)):
        if time_check():
            break
        k1, s1, p1 = pp_for_pairs[i]
        for j in range(i+1, len(pp_for_pairs)):
            k2, s2, p2 = pp_for_pairs[j]
            if p1 == p2:
                continue
            k = k1 * k2
            if k > 10**18:
                break
            M = k * s1 * s2
            if M > 10**18:
                break
            if k not in m_to_ks[M]:
                m_to_ks[M].append(k)
    
    # Triple combinations for small k values
    max_pp_for_triples = min(n_pp, 3000)
    pp_for_triples = prime_powers[:max_pp_for_triples]
    
    for i in range(len(pp_for_triples)):
        if time_check():
            break
        k1, s1, p1 = pp_for_triples[i]
        for j in range(i+1, len(pp_for_triples)):
            k2, s2, p2 = pp_for_triples[j]
            if p1 == p2:
                continue
            k12 = k1 * k2
            if k12 > 10**18:
                continue
            M12 = k12 * s1 * s2
            if M12 > 10**18:
                continue
            for k3, s3, p3 in pp_for_triples:
                if p3 == p1 or p3 == p2:
                    continue
                k = k12 * k3
                if k > 10**18:
                    break
                M = k * s1 * s2 * s3
                if M > 10**18:
                    break
                if k not in m_to_ks[M]:
                    m_to_ks[M].append(k)
    
    # Find M with most k's
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_ks = ks
        if time_check():
            break
    
    # Verify and return
    # Ensure k values are distinct and <= isqrt(M)
    final_ks = []
    for k in best_ks:
        if k <= math.isqrt(best_M):
            final_ks.append(k)
    
    if not final_ks:
        final_ks = [336, 372, 434]
        best_M = 333312
    
    return {'M': int(best_M), 'k': [int(k) for k in final_ks]}
