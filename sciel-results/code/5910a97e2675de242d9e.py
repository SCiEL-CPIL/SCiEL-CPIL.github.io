"""Optimized program for finding M with many solutions to k*sigma(k) = M."""
import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    # Phase 1: Sieve sigma up to N using NumPy for speed
    N = 15000000  # 15 million - larger sieve finds more multi-solution M values
    
    # Compute sigma via NumPy-based sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    if time.time() > deadline - 25:
        # Emergency: return best found so far
        best_M, best_ks = _find_best_from_sigma(sigma, N)
        return _finalize(best_M, best_ks)
    
    # Group k by M_k = k * sigma(k), filter for k <= sqrt(M)
    m_to_k = defaultdict(list)
    sigma_list = sigma.tolist()
    for k in range(1, N + 1):
        M_val = k * sigma_list[k]
        if M_val <= 10**18:
            m_to_k[M_val].append(k)
    
    # Find top candidates by count of valid k (k <= sqrt(M))
    candidates = []
    for M_val, ks in m_to_k.items():
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) >= 2:
            candidates.append((len(valid_ks), M_val, valid_ks))
    
    candidates.sort(key=lambda x: -x[0])
    
    # Start with best candidate
    if candidates:
        best_count, best_M, best_ks = candidates[0]
    else:
        best_count, best_M, best_ks = 0, 333312, [336, 372, 434]
    
    # Phase 2: Extension search for top candidates
    # Check more candidates and deeper extensions
    top_candidates = candidates[:50]
    
    for count, M_val, valid_ks in top_candidates:
        if time.time() > deadline - 10:
            break
        
        sqrt_M = math.isqrt(M_val)
        existing = set(valid_ks)
        ext_ks = list(valid_ks)
        
        # Quick check: only factorize M if it's not too large
        if M_val > 10**16:
            continue  # Too many divisors to check efficiently
        
        # Generate divisors of M_val using factorization
        divisors = _get_divisors(M_val)
        
        # Check divisors k > N where k <= sqrt_M
        for k in divisors:
            if k > N and k <= sqrt_M and k not in existing:
                expected_sigma = M_val // k
                if expected_sigma * k == M_val:  # Verify divisibility
                    if _sigma_by_factorization(k) == expected_sigma:
                        ext_ks.append(k)
                        existing.add(k)
            
            if time.time() > deadline - 8:
                break
        
        if len(ext_ks) > best_count:
            best_M = M_val
            best_ks = sorted(ext_ks)
            best_count = len(ext_ks)
    
    # Phase 3: Additional targeted search - look for M values that might have
    # solutions above N that we missed
    if time.time() < deadline - 5:
        # For the best M found, do a deeper check
        sqrt_M = math.isqrt(best_M)
        existing = set(best_ks)
        
        # Try to find additional solutions by checking divisors more carefully
        divisors = _get_divisors(best_M)
        for k in divisors:
            if k <= sqrt_M and k not in existing:
                expected_sigma = best_M // k
                if expected_sigma * k == best_M:
                    if _sigma_by_factorization(k) == expected_sigma:
                        best_ks.append(k)
                        existing.add(k)
        
        best_ks = sorted(existing)
        best_count = len(best_ks)
    
    return _finalize(best_M, best_ks)

def _find_best_from_map(m_to_k, N):
    """Find best M from map with minimal extension."""
    best_count = 0
    best_M = 333312
    best_ks = [336, 372, 434]
    
    for M_val, ks in m_to_k.items():
        sqrt_M = math.isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > best_count:
            best_M = M_val
            best_ks = sorted(valid_ks)
            best_count = len(valid_ks)
    
    return best_M, best_ks

def _find_best_from_sigma(sigma, N):
    """Find best M directly from sigma array (emergency path)."""
    m_to_k = defaultdict(list)
    sigma_list = sigma.tolist()
    for k in range(1, N + 1):
        M_val = k * sigma_list[k]
        if M_val <= 10**18:
            m_to_k[M_val].append(k)
    return _find_best_from_map(m_to_k, N)

def _finalize(M, ks):
    """Ensure correct types and constraints."""
    M = int(M)
    ks = sorted(set(int(k) for k in ks))
    # Verify k <= isqrt(M)
    sqrt_M = math.isqrt(M)
    ks = [k for k in ks if k <= sqrt_M]
    return {'M': M, 'k': ks}

def _get_divisors(n):
    """Get all divisors of n using factorization. Returns sorted list."""
    if n <= 0:
        return []
    
    factors = _factorize(n)
    
    # Generate all divisors from prime factorization
    divisors = [1]
    for p, e in factors:
        current = list(divisors)
        p_power = 1
        for i in range(e):
            p_power *= p
            for d in current:
                divisors.append(d * p_power)
    
    divisors.sort()
    return divisors

def _factorize(n):
    """Factorize n using trial division with wheel. Returns list of (prime, exponent)."""
    if n <= 1:
        return []
    
    factors = []
    temp = n
    
    # Divide by 2
    if temp % 2 == 0:
        count = 0
        while temp % 2 == 0:
            temp //= 2
            count += 1
        factors.append((2, count))
    
    # Divide by 3
    if temp % 3 == 0:
        count = 0
        while temp % 3 == 0:
            temp //= 3
            count += 1
        factors.append((3, count))
    
    # Wheel: skip multiples of 2 and 3
    d = 5
    wheel = [2, 4, 2, 4, 2, 4, 6, 2, 6]
    wheel_idx = 0
    
    limit = math.isqrt(temp)
    while d <= limit:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            factors.append((d, count))
            limit = math.isqrt(temp)
        
        d += wheel[wheel_idx]
        wheel_idx = (wheel_idx + 1) % len(wheel)
    
    if temp > 1:
        factors.append((temp, 1))
    
    return factors

def _sigma_by_factorization(n):
    """Compute sigma(n) by factoring n."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    factors = _factorize(n)
    result = 1
    for p, e in factors:
        # sigma(p^e) = (p^(e+1) - 1) / (p - 1)
        p_power = p ** (e + 1)
        result *= (p_power - 1) // (p - 1)
    
    return result
