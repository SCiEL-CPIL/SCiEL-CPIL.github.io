import math
import time
import numpy as np

def generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to 'limit'.
    """
    powerful = set()
    max_y = int(math.ceil(limit ** (1.0/3.0))) + 2
    
    sq_free = [True] * (max_y + 1)
    sq_free[0] = False
    p = 2
    while p * p <= max_y:
        if sq_free[p]:
            sq = p * p
            for j in range(sq, max_y + 1, sq):
                sq_free[j] = False
        p += 1
    
    for y in range(1, max_y + 1):
        if not sq_free[y]:
            continue
        y3 = y * y * y
        if y3 > limit:
            break
        max_x = int(math.isqrt(limit // y3))
        for x in range(1, max_x + 1):
            n = x * x * y3
            if n <= limit:
                powerful.add(n)
                
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_n = 0
    best_pairs = []
    
    # Try increasing limits.
    # We use float32 for FFT to reduce memory footprint and potentially speed up.
    # L=10^7 is safe. L=5*10^7 is aggressive.
    limits = [10**7, 5*10**7]
    
    for L in limits:
        if time.time() > deadline - 20:
            break
            
        try:
            P = generate_powerful_numbers(L)
            if not P:
                continue
                
            P_arr = np.array(P, dtype=np.int64)
            P_set = set(P)
            
            # FFT Convolution with float32
            target_size = 2 * L + 1
            n_fft = 1
            while n_fft < target_size:
                n_fft *= 2
                
            # Check memory: n_fft * 4 bytes (float32) * 2 (complex) * 2 (real/imag storage in rfft? rfft is n_fft//2+1 complex)
            # rfft output size is n_fft//2 + 1 complex128.
            # If we use float32 input, rfft still outputs float64/complex128 by default in numpy?
            # Actually, numpy.fft.rfft promotes to float64.
            # To keep it float32, we might need scipy or manual handling, but let's assume standard numpy.
            # If standard numpy promotes, memory is 2x.
            # Let's stick to float64 but be careful with L.
            
            if n_fft > 2**26: # ~67M points, ~500MB for complex128
                # If L=5*10^7, n_fft = 2^27 = 134M. This is ~1GB.
                # Might be too much. Let's cap at 2**26 for safety unless we know memory is huge.
                # But the prompt says "NumPy and SciPy are available".
                # Let's try L=5*10^7 only if we can fit it.
                if L > 2*10**7:
                    continue
            
            A = np.zeros(n_fft, dtype=np.float64)
            A[P_arr[P_arr < n_fft]] = 1.0
            
            F = np.fft.rfft(A)
            F2 = F * F
            result = np.fft.irfft(F2, n=n_fft)
            
            scores = result[:2*L+1]
            
            # Calculate d
            d = np.zeros(2*L+1, dtype=np.int32)
            even_indices = np.arange(2, 2*L+1, 2)
            half_indices = even_indices // 2
            P_bool = np.zeros(L+1, dtype=np.int8)
            P_bool[P_arr[P_arr <= L]] = 1
            d[even_indices] = P_bool[half_indices]
            
            scores_int = np.rint(scores).astype(np.int64)
            total_pairs = (scores_int + d) // 2
            
            # Find top candidates
            if len(total_pairs) > 100:
                top_indices = np.argpartition(total_pairs, -100)[-100:]
            else:
                top_indices = np.arange(len(total_pairs))
                
            top_indices = top_indices[np.argsort(total_pairs[top_indices])[::-1]]
            
            current_best_n = 0
            current_best_pairs = []
            
            for idx in top_indices:
                N = int(idx)
                if N < 2:
                    continue
                
                pairs = []
                for a in P:
                    if a > N // 2:
                        break
                    b = N - a
                    if b in P_set:
                        pairs.append([a, b])
                        
                if len(pairs) > len(current_best_pairs):
                    current_best_pairs = pairs
                    current_best_n = N
                    
                # Early stop if we found a very high count?
                # Max possible is small.
                
            if len(current_best_pairs) > len(best_pairs):
                best_pairs = current_best_pairs
                best_n = current_best_n
                
        except Exception as e:
            # Fallback to smaller L if memory/time fails
            continue
            
    # Fallback
    if not best_pairs:
        best_n = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_n, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
