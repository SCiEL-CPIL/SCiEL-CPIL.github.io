"""
Find N with many representations as sums of two powerful numbers.
Strategy:
  1. Generate a large set of powerful numbers up to ~1e9.
  2. Construct candidate N values from sums of "rich" powerful numbers
     (those with many prime factors / high divisor count).
  3. Verify each candidate N using numpy-vectorized membership checks.
  4. Local search around the best N found.
"""
import time
import numpy as np
from math import isqrt
from collections import defaultdict

def _sieve_primes(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _generate_powerful(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number n can be written as m^2 * b where b is squarefree
    and every prime dividing b also divides m.
    Equivalently, n = prod p_i^{e_i} with e_i >= 2.
    We iterate m from 1 to sqrt(limit), factor m, and for each squarefree
    divisor b of m's prime set, add m^2 * b if <= limit.
    """
    sqrt_limit = isqrt(limit)
    powerful_set = set([1])
    
    # Precompute smallest prime factor up to sqrt_limit for fast factorization
    spf = list(range(sqrt_limit + 1))
    for i in range(2, isqrt(sqrt_limit) + 1):
        if spf[i] == i:
            for j in range(i * i, sqrt_limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for m in range(2, sqrt_limit + 1):
        # Factor m to get distinct prime factors
        temp = m
        primes = []
        while temp > 1:
            if temp <= sqrt_limit:
                p = spf[temp]
            else:
                p = temp
            primes.append(p)
            while temp % p == 0:
                temp //= p
        
        # Enumerate all subsets of distinct primes -> squarefree divisors
        n_primes = len(primes)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= primes[i]
            m2 = m * m
            if b <= limit // m2:
                val = m2 * b
                if val <= limit:
                    powerful_set.add(val)
    
    return sorted(powerful_set)

def _divisor_count(n, spf, spf_limit):
    """Count divisors of n using precomputed SPF."""
    if n == 1:
        return 1
    count = 1
    temp = n
    while temp > 1:
        if temp <= spf_limit:
            p = spf[temp]
        else:
            p = temp
        exp = 0
        while temp % p == 0:
            temp //= p
            exp += 1
        count *= (exp + 1)
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Phase 1: Generate powerful numbers
    PW_LIMIT = 1_000_000_000  # 10^9
    powerful = _generate_powerful(PW_LIMIT)
    powerful_set = set(powerful)
    P = len(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 80:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    pw_arr = np.array(powerful, dtype=np.int64)
    
    # Phase 2: Score powerful numbers by "richness" (divisor count)
    # Precompute SPF for divisor counting
    spf_limit = isqrt(PW_LIMIT) + 2
    spf = list(range(spf_limit + 1))
    for i in range(2, isqrt(spf_limit) + 1):
        if spf[i] == i:
            for j in range(i * i, spf_limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    scored = []
    for p in powerful:
        if p > 1:
            d = _divisor_count(p, spf, spf_limit)
            scored.append((d, p))
    
    scored.sort(reverse=True)
    top_rich = [p for _, p in scored[:500]]
    
    # Phase 3: Generate candidate N values
    candidates = set()
    
    # Sums of pairs of rich powerful numbers
    for i in range(len(top_rich)):
        for j in range(i, len(top_rich)):
            N_val = top_rich[i] + top_rich[j]
            if N_val <= 10**18:
                candidates.add(N_val)
    
    # Also: 2 * p for top powerful (a = b case)
    for p in top_rich[:200]:
        if 2 * p <= 10**18:
            candidates.add(2 * p)
    
    # Also: sums involving 1 (since 1 is powerful)
    for p in top_rich[:300]:
        candidates.add(p + 1)
    
    # Also: try N values that are sums of a rich powerful and a small powerful
    small_pw = powerful[:200]
    for p_rich in top_rich[:100]:
        for p_small in small_pw:
            N_val = p_rich + p_small
            if N_val <= 10**18:
                candidates.add(N_val)
    
    candidates = sorted(candidates)
    
    # Phase 4: Verify candidates using numpy
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    for N in candidates:
        if time.time() > deadline - 20:
            break
        
        # a must satisfy: 1 <= a <= N/2, a in powerful, N - a in powerful
        # Use numpy for the a <= N/2 filter
        half_N = N // 2
        mask_a = pw_arr <= half_N
        a_candidates = pw_arr[mask_a]
        
        if len(a_candidates) == 0:
            continue
        
        b_values = N - a_candidates
        # Filter: b must be >= a (i.e., b >= 1, which is automatic since b = N - a >= N - N/2 = N/2 >= a)
        # Actually b >= a means N - a >= a means a <= N/2, already enforced.
        # b must be in powerful_set and b >= 1
        valid_mask = (b_values >= 1)
        a_valid = a_candidates[valid_mask]
        b_valid = b_values[valid_mask]
        
        if len(a_valid) == 0:
            continue
        
        # Check membership in powerful_set
        # For efficiency, convert to Python list for set lookup
        a_list = a_valid.tolist()
        b_list = b_valid.tolist()
        
        # Use set intersection for speed
        b_set = set(b_list)
        a_in_set = [a for a in a_list if a in powerful_set]
        # But we also need b = N - a to be in powerful_set
        # Since a is already from powerful array, a is powerful.
        # We just need to check if N - a is in powerful_set.
        valid_a = [a for a in a_list if (N - a) in powerful_set and (N - a) >= a]
        
        count = len(valid_a)
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = [[a, N - a] for a in valid_a]
    
    # Phase 5: Local search around best N
    if time.time() < deadline - 25 and best_N > 2:
        # Search in a window around best_N
        window = 500
        for delta in range(1, window + 1):
            for N_try in [best_N + delta, best_N - delta]:
                if N_try < 2 or N_try > 10**18:
                    continue
                if time.time() > deadline - 15:
                    break
                
                half_N = N_try // 2
                mask_a = pw_arr <= half_N
                a_candidates = pw_arr[mask_a]
                
                if len(a_candidates) == 0:
                    continue
                
                a_list = a_candidates.tolist()
                valid_a = [a for a in a_list if (N_try - a) in powerful_set and (N_try - a) >= a]
                
                count = len(valid_a)
                if count > best_count:
                    best_count = count
                    best_N = N_try
                    best_pairs = [[a, N_try - a] for a in valid_a]
    
    # Phase 6: Broader local search if time remains
    if time.time() < deadline - 10 and best_N > 2:
        # Try a wider but sparser search
        step = 10
        for delta in range(step, 5000, step):
            for N_try in [best_N + delta, best_N - delta]:
                if N_try < 2 or N_try > 10**18:
                    continue
                if time.time() > deadline - 5:
                    break
                
                half_N = N_try // 2
                mask_a = pw_arr <= half_N
                a_candidates = pw_arr[mask_a]
                
                if len(a_candidates) == 0:
                    continue
                
                a_list = a_candidates.tolist()
                valid_a = [a for a in a_list if (N_try - a) in powerful_set and (N_try - a) >= a]
                
                count = len(valid_a)
                if count > best_count:
                    best_count = count
                    best_N = N_try
                    best_pairs = [[a, N_try - a] for a in valid_a]
    
    return {'N': best_N, 'pairs': best_pairs}


if __name__ == "__main__":
    result = construct()
    print(f"N = {result['N']}")
    print(f"Number of pairs: {len(result['pairs'])}")
    print(f"First few pairs: {result['pairs'][:10]}")
