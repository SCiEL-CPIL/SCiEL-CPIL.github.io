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
    
    primes = sieve(200000)
    
    # Map M -> list of k
    m_to_ks = defaultdict(list)
    
    # Process single prime powers: k = p^a
    # k * sigma(k) = p^a * (p^(a+1)-1)/(p-1)
    for p in primes:
        if time_check():
            break
        pk = p  # p^1
        # sigma(p^a) = (p^(a+1)-1)/(p-1)
        # We need pk * sigma(pk) <= 10^18
        pk_val = p
        sigma_val = p + 1  # sigma(p^1) = 1 + p
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 2: k = p^2
        pk_val = p * p
        if pk_val > 10**18:
            break
        sigma_val = 1 + p + p*p
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 3: k = p^3
        pk_val = p * p * p
        if pk_val > 10**18:
            break
        sigma_val = 1 + p + p*p + p*p*p
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 4
        pk_val = p**4
        if pk_val > 10**18:
            break
        sigma_val = 1 + p + p*p + p**3 + p**4
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 5
        pk_val = p**5
        if pk_val > 10**18:
            break
        sigma_val = 1 + p + p*p + p**3 + p**4 + p**5
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 6
        pk_val = p**6
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(7):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 7
        pk_val = p**7
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(8):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 8
        pk_val = p**8
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(9):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 9
        pk_val = p**9
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(10):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 10
        pk_val = p**10
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(11):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 11
        pk_val = p**11
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(12):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 12
        pk_val = p**12
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(13):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 13
        pk_val = p**13
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(14):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 14
        pk_val = p**14
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(15):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 15
        pk_val = p**15
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(16):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 16
        pk_val = p**16
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(17):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 17
        pk_val = p**17
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(18):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 18
        pk_val = p**18
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(19):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 19
        pk_val = p**19
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(20):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
        
        # a = 20
        pk_val = p**20
        if pk_val > 10**18:
            break
        sigma_val = 1
        pk_temp = 1
        for i in range(21):
            pk_temp *= p
            sigma_val += pk_temp
        M_val = pk_val * sigma_val
        if M_val <= 10**18:
            if pk_val not in m_to_ks[M_val]:
                m_to_ks[M_val].append(pk_val)
    
    # Process products of two distinct prime powers: k = p^a * q^b
    # k * sigma(k) = p^a * q^b * sigma(p^a) * sigma(q^b)
    # Build list of (k_part, sigma_part) for each prime power
    prime_powers = []  # list of (k_val, sigma_val, p)
    for p in primes:
        if time_check():
            break
        pk_val = p
        sigma_val = 1 + p
        if pk_val * sigma_val <= 10**18:
            prime_powers.append((pk_val, sigma_val, p))
        
        pk_val = p * p
        sigma_val = 1 + p + p*p
        if pk_val * sigma_val <= 10**18:
            prime_powers.append((pk_val, sigma_val, p))
        
        pk_val = p**3
        if pk_val > 10**18:
            break
        sigma_val = 1 + p + p*p + p**3
        if pk_val * sigma_val <= 10**18:
            prime_powers.append((pk_val, sigma_val, p))
        
        pk_val = p**4
        if pk_val > 10**18:
            break
        sigma_val = 1 + p + p*p + p**3 + p**4
        if pk_val * sigma_val <= 10**18:
            prime_powers.append((pk_val, sigma_val, p))
        
        pk_val = p**5
        if pk_val > 10**18:
            break
        sigma_val = 1
        t = 1
        for i in range(6):
            t *= p
            sigma_val += t
        if pk_val * sigma_val <= 10**18:
            prime_powers.append((pk_val, sigma_val, p))
        
        pk_val = p**6
        if pk_val > 10**18:
            break
        sigma_val = 1
        t = 1
        for i in range(7):
            t *= p
            sigma_val += t
        if pk_val * sigma_val <= 10**18:
            prime_powers.append((pk_val, sigma_val, p))
    
    # Now combine pairs of prime powers with different primes
    # To keep it manageable, limit the number of combinations
    n_pp = len(prime_powers)
    if time_check():
        pass  # skip pair combinations if out of time
    
    # For pair combinations, we need to be selective
    # Only combine if the product k1*k2 * sigma1*sigma2 <= 10^18
    # And k1*k2 <= 10^18 (which is implied since k*sigma(k) >= k^2 for k>=2)
    
    # To limit work, only consider pairs where both k values are small
    # Sort prime_powers by k value
    prime_powers.sort(key=lambda x: x[0])
    
    # Limit to first N prime powers for pairing to control complexity
    max_pp_for_pairs = min(n_pp, 5000)
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
