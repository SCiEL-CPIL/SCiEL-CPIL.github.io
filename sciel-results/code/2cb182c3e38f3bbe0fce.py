import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin from 150s limit
    
    # Try increasing L values
    # L=10^8 -> 2L=2*10^8 -> FFT size 2^28 (~2.68e8)
    # float32 FFT memory: ~1GB for A, ~1GB for F, ~1GB for C. Total ~3-4GB.
    # This is risky but worth trying.
    
    L_candidates = [10 * 10**7, 5 * 10**7, 3 * 10**7]
    
    best_result = None
    
    for L in L_candidates:
        try:
            if time.time() > deadline - 30:
                continue # Not enough time for generation + FFT
            
            # 1. Generate powerful numbers <= L
            powerful_set = set()
            max_y = int(L ** (1/3)) + 1
            
            # Sieve for square-free numbers up to max_y
            # max_y is ~464 for 1e8
            is_sq_free = [True] * (max_y + 1)
            is_sq_free[0] = False
            for i in range(2, int(math.isqrt(max_y)) + 1):
                if is_sq_free[i]:
                    for j in range(i*i, max_y + 1, i*i):
                        is_sq_free[j] = False
            
            sq_free = [y for y in range(1, max_y + 1) if is_sq_free[y]]
            
            for y in sq_free:
                y_cubed = y ** 3
                if y_cubed > L:
                    continue
                max_x = int(math.isqrt(L // y_cubed))
                for x in range(1, max_x + 1):
                    val = x * x * y_cubed
                    powerful_set.add(val)
            
            if time.time() > deadline - 20:
                continue # Not enough time for FFT
            
            P = sorted(powerful_set)
            
            # 2. FFT Convolution
            n_fft = 1
            while n_fft < 2 * L + 1:
                n_fft *= 2
                
            # Check memory feasibility
            # For L=1e8, n_fft=2^28=268,435,456
            # float32 array size: 268e6 * 4 bytes = 1.07 GB
            # complex64 array size: 134e6 * 8 bytes = 1.07 GB
            # Total ~ 3-4 GB.
            
            # Use float32 to save memory
            A = np.zeros(n_fft, dtype=np.float32)
            for p in P:
                A[p] = 1.0
                
            F = np.fft.rfft(A)
            C = np.fft.irfft(F * F, n=n_fft)
            
            # C is float32. Round to int64
            # Values are small counts, so rounding is safe
            C_int = np.round(C).astype(np.int64)
            
            # 3. Find max count
            max_idx = np.argmax(C_int)
            best_N = int(max_idx)
            
            if best_N < 2 or best_N > 2 * L:
                best_N = 2
                
            # 4. Retrieve pairs
            pairs = []
            powerful_set_lookup = powerful_set
            half_best = best_N // 2
            
            for a in P:
                if a > half_best:
                    break
                b = best_N - a
                if b in powerful_set_lookup:
                    pairs.append([a, b])
            
            # Update best result
            # We want to maximize the number of pairs
            # Since we try larger L first, and larger L allows more powerful numbers,
            # the max count found in a larger L range is likely >= smaller L range?
            # Not necessarily, the specific N with max count might be in a smaller range.
            # But usually, more numbers -> higher peak.
            # We keep the one with the most pairs.
            if best_result is None or len(pairs) > len(best_result['pairs']):
                best_result = {'N': best_N, 'pairs': pairs}
                
            # If we found a good number of pairs, we might stop?
            # No, larger L might have even more.
            # But if time is tight, we should prioritize the largest L that fits.
            
        except MemoryError:
            continue
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to smallest L
        L = 2 * 10**7
        powerful_set = set()
        max_y = int(L ** (1/3)) + 1
        is_sq_free = [True] * (max_y + 1)
        is_sq_free[0] = False
        for i in range(2, int(math.isqrt(max_y)) + 1):
            if is_sq_free[i]:
                for j in range(i*i, max_y + 1, i*i):
                    is_sq_free[j] = False
        sq_free = [y for y in range(1, max_y + 1) if is_sq_free[y]]
        
        for y in sq_free:
            y_cubed = y ** 3
            if y_cubed > L:
                continue
            max_x = int(math.isqrt(L // y_cubed))
            for x in range(1, max_x + 1):
                val = x * x * y_cubed
                powerful_set.add(val)
                
        P = sorted(powerful_set)
        n_fft = 1
        while n_fft < 2 * L + 1:
            n_fft *= 2
            
        A = np.zeros(n_fft, dtype=np.float64)
        for p in P:
            A[p] = 1.0
            
        F = np.fft.rfft(A)
        C = np.fft.irfft(F * F, n=n_fft)
        C_int = np.round(C).astype(np.int64)
        
        max_idx = np.argmax(C_int)
        best_N = int(max_idx)
        
        pairs = []
        powerful_set_lookup = powerful_set
        half_best = best_N // 2
        
        for a in P:
            if a > half_best:
                break
            b = best_N - a
            if b in powerful_set_lookup:
                pairs.append([a, b])
                
        return {'N': best_N, 'pairs': pairs}
        
    return best_result
