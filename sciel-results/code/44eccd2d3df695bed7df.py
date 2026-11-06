import time
import math
import numpy as np
from bisect import bisect_left

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    powerful = set()
    powerful.add(1)
    
    max_b = int(round(limit ** (1/3))) + 2
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 15s safety margin
    deadline = start_time + 135.0
    
    # Phase 1: Generate powerful numbers up to a large limit
    # Try to go higher than 100M to find better candidates
    # 500M powerful numbers -> ~22K numbers. FFT size 1B -> 4GB float32.
    # Let's try 200M first to be safe, then 500M if time permits.
    
    limits_to_try = [500_000_000, 200_000_000, 100_000_000]
    
    best_result = {'N': 2, 'pairs': [[1, 1]], 'count': 1}
    
    for limit in limits_to_try:
        if time.time() > deadline - 60:
            break
            
        try:
            powerful = _generate_powerful(limit)
            powerful_set = set(powerful)
            K = len(powerful)
            
            if K < 2:
                continue
            
            # Phase 2: FFT Convolution
            max_sum = 2 * limit
            n_fft = 1
            while n_fft < max_sum + 1:
                n_fft *= 2
            
            # Check memory: n_fft * 4 bytes (float32)
            # 1B * 4 = 4GB. Might be tight.
            # If limit=500M, max_sum=1B, n_fft=1B (2^30).
            # 2^30 * 4 bytes = 4GB.
            # If limit=200M, max_sum=400M, n_fft=512M (2^29).
            # 2^29 * 4 bytes = 2GB. Safer.
            
            if n_fft * 4 > 4 * 1024**3: # 4GB limit
                continue
                
            f = np.zeros(n_fft, dtype=np.float32)
            for val in powerful:
                if val < n_fft:
                    f[val] = 1.0
            
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=n_fft)
            
            conv_int = np.round(conv[:max_sum + 1]).astype(np.int64)
            
            # Compute pairs count
            pairs_count = np.zeros(max_sum + 1, dtype=np.int64)
            even_s = np.arange(0, max_sum + 1, 2)
            half_s = even_s // 2
            is_powerful_half = (f[half_s] == 1.0)
            
            adjusted = conv_int.copy()
            adjusted[even_s[is_powerful_half]] += 1
            pairs_count = adjusted // 2
            
            if max_sum >= 2:
                pairs_count[:2] = 0
            
            # Find top candidates
            top_k = 50
            if len(pairs_count) > top_k:
                top_indices = np.argpartition(pairs_count, -top_k)[-top_k:]
            else:
                top_indices = np.arange(len(pairs_count))
                
            top_indices = top_indices[np.argsort(pairs_count[top_indices])[::-1]]
            
            # Verify
            for idx in top_indices:
                if time.time() > deadline - 10:
                    break
                    
                N = int(idx)
                if N < 2:
                    continue
                
                count = 0
                exact_pairs = []
                for a in powerful:
                    if a > N // 2:
                        break
                    b = N - a
                    if b in powerful_set:
                        count += 1
                        exact_pairs.append([a, b])
                
                if count > best_result['count']:
                    best_result['count'] = count
                    best_result['N'] = N
                    best_result['pairs'] = exact_pairs
                    
                if pairs_count[idx] < best_result['count']:
                    break
                    
        except MemoryError:
            continue
        except Exception as e:
            continue
            
    return {'N': best_result['N'], 'pairs': best_result['pairs']}
