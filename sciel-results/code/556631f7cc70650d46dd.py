import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to use a larger L to find N with more representations
    L_candidates = [10**9, 5 * 10**8, 10**8, 5 * 10**7]
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_score = 1
    
    for L in L_candidates:
        if time.time() > deadline - 30:
            break
            
        # Generate powerful numbers up to L
        powerfuls = set()
        max_b = int(L ** (1.0/3.0)) + 2
        for b in range(1, max_b + 1):
            b3 = b * b * b
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerfuls.add(n)
        
        powerfuls_list = sorted(powerfuls)
        M = len(powerfuls_list)
        
        if M < 10:
            continue

        # Prepare FFT
        n_fft = 1
        while n_fft < 2 * L + 1:
            n_fft *= 2
            
        try:
            A = np.zeros(n_fft, dtype=np.float64)
            A[powerfuls_list] = 1.0
        except MemoryError:
            continue
            
        try:
            F = np.fft.rfft(A)
            C = F * F
            conv = np.fft.irfft(C, n=n_fft)
        except Exception:
            continue
            
        if time.time() > deadline - 10:
            continue

        conv_int = np.rint(conv[:2*L+1]).astype(np.int64)
        
        if len(conv_int) < 3:
            continue
            
        max_count = int(np.max(conv_int[2:]))
        best_N = int(np.argmax(conv_int[2:])) + 2
        
        if max_count > best_score:
            # Extract pairs
            powerfuls_set = set(powerfuls_list)
            pairs = []
            for a in powerfuls_list:
                if a > best_N // 2:
                    break
                b = best_N - a
                if b in powerfuls_set:
                    pairs.append([a, b])
                    
            pairs.sort()
            
            if pairs:
                best_score = max_count
                best_result = {'N': best_N, 'pairs': pairs}
                
        # If we successfully processed a large L, we likely have a good result.
        # Smaller L will likely yield smaller max counts.
        # We can break if we are confident, but let's just let the loop finish or break on time.
        # Actually, if 10^9 works, it's the best bet.
        if L == 10**9:
             break

    return best_result
