import time
import numpy as np
from math import isqrt

def _is_square_free(n):
    """Check if n is square-free."""
    if n < 2:
        return True
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    r = isqrt(n)
    if r * r == n:
        return False
    c = round(n ** (1/3))
    for cc in range(max(2, c-2), c+3):
        if cc**3 == n:
            return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit."""
    powerful = set([1])
    max_b = int(limit ** (1/3)) + 2
    
    sq_free = [True] * (max_b + 1)
    sq_free[0] = False
    if max_b >= 1:
        sq_free[1] = True
    
    for p in range(2, int(max_b ** 0.5) + 2):
        if sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                sq_free[k] = False
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def _count_reps_for_n(n, powerful, powerful_set):
    """Count and collect pairs (a,b) with a<=b, a+b=n, both powerful."""
    pairs = []
    for a in powerful:
        if 2 * a > n:
            break
        b = n - a
        if b in powerful_set:
            pairs.append([a, b])
    return pairs

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer before 150s limit
    
    # Strategy: Use FFT to find N with many representations among powerful numbers
    # up to a limit, then verify and collect pairs.
    
    # Try progressively larger limits if time permits
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_score = 1
    
    # Start with a moderate limit that should fit in time
    limit_pw = 30_000_000
    
    try:
        powerful = _generate_powerful(limit_pw)
        powerful_set = set(powerful)
        
        elapsed_gen = time.time() - start_time
        remaining = deadline - time.time()
        
        if remaining > 20:
            # Create binary array
            size = 2 * limit_pw + 10
            pw_array = np.zeros(size, dtype=np.float64)
            for p in powerful:
                if p < size:
                    pw_array[p] = 1.0
            
            # FFT size
            fft_size = 1 << (size - 1).bit_length()
            if fft_size < size:
                fft_size = fft_size * 2
            
            f = np.fft.rfft(pw_array, n=fft_size)
            power = f * np.conj(f)
            inv = np.fft.irfft(power, n=fft_size)
            
            max_sum = 2 * limit_pw
            inv_counts = inv[:max_sum + 1]
            
            # Find top candidates
            # Get top 10 indices by count
            top_indices = np.argsort(inv_counts)[::-1][:10]
            
            for idx in top_indices:
                n_candidate = int(idx)
                if n_candidate < 2 or n_candidate > 10**18:
                    continue
                
                pairs = _count_reps_for_n(n_candidate, powerful, powerful_set)
                if len(pairs) > best_score:
                    best_score = len(pairs)
                    best_result = {'N': n_candidate, 'pairs': pairs}
                
                if time.time() > deadline - 10:
                    break
            
    except Exception:
        pass
    
    # If we have time, try a larger limit
    if time.time() < deadline - 30 and best_score < 10:
        try:
            limit_pw2 = 50_000_000
            powerful2 = _generate_powerful(limit_pw2)
            powerful_set2 = set(powerful2)
            
            size2 = 2 * limit_pw2 + 10
            pw_array2 = np.zeros(size2, dtype=np.float64)
            for p in powerful2:
                if p < size2:
                    pw_array2[p] = 1.0
            
            fft_size2 = 1 << (size2 - 1).bit_length()
            if fft_size2 < size2:
                fft_size2 = fft_size2 * 2
            
            f2 = np.fft.rfft(pw_array2, n=fft_size2)
            power2 = f2 * np.conj(f2)
            inv2 = np.fft.irfft(power2, n=fft_size2)
            
            max_sum2 = 2 * limit_pw2
            inv_counts2 = inv2[:max_sum2 + 1]
            
            top_indices2 = np.argsort(inv_counts2)[::-1][:10]
            
            for idx in top_indices2:
                n_candidate = int(idx)
                if n_candidate < 2 or n_candidate > 10**18:
                    continue
                
                pairs = _count_reps_for_n(n_candidate, powerful2, powerful_set2)
                if len(pairs) > best_score:
                    best_score = len(pairs)
                    best_result = {'N': n_candidate, 'pairs': pairs}
                
                if time.time() > deadline - 10:
                    break
                    
        except Exception:
            pass
    
    return best_result
