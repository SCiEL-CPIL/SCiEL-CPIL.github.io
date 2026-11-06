"""
Strategy: FFT-based convolution for powerful number pair counting.
Instead of O(n^2) pairwise summation, we construct a binary array of powerful numbers
up to a limit L, compute the autocorrelation (via FFT) to find N with many representations,
and then verify/extract pairs for the top candidates.
This allows searching N up to ~10^9 efficiently.
"""

# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _generate_powerfuls(limit):
    """
    Generate all powerful numbers <= limit.
    A powerful number has the form a^2 * b^3 where b is square-free.
    """
    powerfuls = set()
    # b must be square-free and b^3 <= limit
    max_b = int(limit ** (1/3)) + 2
    
    # Primes up to sqrt(max_b) for square-free check
    sqrt_max_b = isqrt(max_b)
    primes = []
    for x in range(2, sqrt_max_b + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        is_sq_free = True
        for p in primes:
            if p * p > b:
                break
            if b % (p * p) == 0:
                is_sq_free = False
                break
        if not is_sq_free:
            continue
        
        b3 = b ** 3
        if b3 > limit:
            break
        
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    
    return powerfuls

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # We want to find N up to some max_N.
    # If N = a + b, and a,b <= L, then N <= 2L.
    # The number of powerful numbers up to L is approx L^0.5 * (some constant).
    # For L = 5*10^7, N up to 10^8.
    # For L = 2.5*10^8, N up to 5*10^8.
    
    # FFT size needs to be power of 2, >= 2*L.
    # Let's aim for L = 2 * 10^8 (N up to 4 * 10^8).
    # FFT size 2^29 = 536,870,912. This is large but manageable in memory (complex128 is 16 bytes -> ~8GB? No, 500M * 16 = 8GB. Too much for standard RAM maybe? 
    # Let's check typical limits. 512M complex numbers is 8GB. 
    # Let's try a smaller L first to be safe, then scale up if time permits.
    # L = 5 * 10^7 -> N up to 10^8. FFT size 2^27 = 134M. 134M * 16 bytes = 2.1GB. This is safer.
    
    # Actually, let's dynamically pick L based on available time and memory heuristics.
    # Let's start with L = 10^8 (N up to 2*10^8). FFT size 2^28 = 268M. ~4GB. 
    # Python/NumPy might struggle with 4GB allocation in some environments.
    # Let's try L = 5*10^7 first.
    
    L = 50_000_000  # 5 * 10^7
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback for very slow systems
        L = 10_000_000

    powerfuls = _generate_powerfuls(L)
    powerfuls_list = sorted(powerfuls)
    powerfuls_set = set(powerfuls_list)
    
    if time.time() > deadline - 20:
        # If generation was slow, fallback to small brute force
        # Just return a small known good or simple pair
        # 1 is powerful. 1+1=2.
        # Let's try to find a small N with a few pairs quickly
        # 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 108, 121, 125, 128, 144, 169, 196, 200, 216, 225, 243, 256, 289, 324, 343, 361, 400, 441, 484, 500, 512, 529, 576, 625, 675, 676, 729, 784, 800, 841, 900, 961, 968, 1000, 1024...
        # 1+8=9, 4+5(no), 9+0(no)
        # 1+15(no), 4+12(no), 8+8=16 (1,8,8,16 all powerful? 1=yes, 8=2^3 yes, 16=2^4 yes)
        # So N=16 has pair (1,15 no), (4,12 no), (8,8 yes). 
        # N=9: (1,8) yes.
        # N=17: (1,16) yes.
        # N=25: (1,24 no), (9,16) yes.
        # N=29: (1,28 no), (9,20 no), (16,13 no), (25,4) yes.
        # N=33: (1,32) yes.
        # N=37: (1,36) yes, (9,28 no), (16,21 no), (25,12 no), (32,5 no).
        # N=41: (1,40 no), (9,32) yes.
        # N=45: (1,44 no), (9,36) yes.
        # N=49: (1,48 no), (9,40 no), (16,33 no), (25,24 no), (36,13 no), (49,0 no). (1,48) no. 
        # Wait, 49 is powerful. 0 is not.
        # N=53: (1,52 no), (9,44 no), (16,37 no), (25,28 no), (36,17 no), (49,4) yes.
        
        # Let's just do a quick brute force on small range
        small_limit = 1000
        small_pows = [p for p in powerfuls if p <= small_limit]
        count_dict = {}
        pair_dict = {}
        for i in range(len(small_pows)):
            a = small_pows[i]
            for j in range(i, len(small_pows)):
                b = small_pows[j]
                s = a + b
                if s > small_limit * 2: break
                if s not in count_dict:
                    count_dict[s] = 0
                    pair_dict[s] = []
                count_dict[s] += 1
                pair_dict[s].append([a, b])
        
        if count_dict:
            best_N = max(count_dict, key=count_dict.get)
            return {'N': best_N, 'pairs': pair_dict[best_N]}
        return {'N': 2, 'pairs': [[1, 1]]}

    # Create indicator array
    # Size needs to be at least 2*L for the convolution to capture all sums up to 2L.
    # We use FFT size which is next power of 2 >= 2*L.
    size = 1
    while size < 2 * L:
        size *= 2
        
    # If size is too large, reduce L
    # 2^28 = 268M. 2^29 = 536M.
    # Let's cap at 2^27 (134M) to be safe with memory (2GB).
    max_fft_size = 1 << 27
    if size > max_fft_size:
        # Reduce L
        L = max_fft_size // 2
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        size = max_fft_size
        if time.time() > deadline - 15:
             return {'N': 2, 'pairs': [[1, 1]]}

    arr = np.zeros(size, dtype=np.float64)
    for p in powerfuls_list:
        if p < size:
            arr[p] = 1.0
            
    # Compute autocorrelation via FFT
    # FFT(arr) * conj(FFT(arr)) -> IFFT -> counts
    # Note: IFFT of |FFT|^2 gives circular autocorrelation.
    # Since size >= 2*max_val, linear autocorrelation is embedded in the first L elements (and last L-1).
    # Specifically, r[k] = sum_i arr[i] * arr[i+k].
    # We want count of a+b=N. This is sum_{a+b=N} arr[a]arr[b].
    # This is exactly the convolution of arr with arr reversed? 
    # No, it's the cross-correlation.
    # Let A = FFT(arr).
    # Convolution c = IFFT(A * A) gives sum arr[i]arr[j] where i+j = k (mod size).
    # Since we are looking for a+b=N, and a,b < L < size/2, the wrap-around doesn't affect indices < L.
    # So c[N] = number of pairs (a,b) such that a+b=N.
    
    try:
        fft_arr = np.fft.fft(arr)
        power_spec = fft_arr * fft_arr  # Note: for real input, we can use rfft for speed, but fft is fine.
        # Actually, for autocorrelation of real signal, we use |FFT|^2.
        # But convolution of f with f is IFFT(FFT(f)^2).
        # We want sum_{a+b=N} f(a)f(b). This is the N-th element of the convolution of f with f.
        # So yes, IFFT(FFT(f)^2).
        
        conv = np.fft.ifft(power_spec).real
        # conv[k] is the count of pairs summing to k.
        
        # Find the max count
        # We only care about N <= 2*L
        # The counts for N > 2*L are 0 (or wrap-around if size < 2L, but we ensured size >= 2L).
        # Actually, since size is power of 2 and >= 2L, the linear convolution is exactly in conv[0:size].
        
        # We want to maximize the count.
        # The count includes (a,b) and (b,a).
        # The task asks for distinct pairs a <= b.
        # If a != b, the convolution counts both (a,b) and (b,a).
        # If a == b, it counts (a,a) once.
        # So count_conv[N] = 2 * (# pairs with a < b) + (# pairs with a = b).
        # We want to maximize # pairs with a <= b.
        # This is (count_conv[N] + (1 if N is even and N/2 is powerful else 0)) / 2.
        # To find the best N, we can just look for max count_conv[N], 
        # but we must be careful: a high count_conv might be due to many distinct pairs.
        # Generally, maximizing count_conv is a good proxy.
        
        # Let's find top candidates
        # conv is an array of size `size`.
        # We only need to look up to 2*L.
        search_limit = min(2 * L, size)
        
        # Find indices of top counts
        # To avoid memory issues with argsort on large array, we can use partition or just max.
        # But we might want top 10 to verify.
        
        # Let's just find the global max in the valid range
        valid_conv = conv[:search_limit]
        
        # Find the index of the maximum
        best_N_idx = int(np.argmax(valid_conv))
        best_count_conv = valid_conv[best_N_idx]
        
        # It's possible that a slightly lower conv count yields a higher "distinct pairs" score?
        # No, the mapping is monotonic for fixed parity of diagonal.
        # Let's check top 5 candidates to be safe.
        
        # Get top 5 indices
        # For large arrays, argsort is O(N log N). 
        # We can use np.partition to get top 5 in O(N).
        if search_limit > 1000:
            top_k = 5
            indices = np.argpartition(valid_conv, -top_k)[-top_k:]
        else:
            indices = np.argsort(valid_conv)[::-1][:5]
        
        best_N = None
        best_pairs = []
        best_score = 0
        
        powerfuls_set_local = powerfuls_set
        powerfuls_list_local = powerfuls_list
        
        for idx in indices:
            N = int(idx)
            if N < 2:
                continue
            
            # Extract pairs
            pairs = []
            # a <= b => a <= N/2
            # a must be powerful
            # b = N - a must be powerful
            # Iterate through powerfuls <= N/2
            half_N = N // 2
            # Find end index in powerfuls_list
            # powerfuls_list is sorted
            # Use bisect
            import bisect
            end_idx = bisect.bisect_right(powerfuls_list_local, half_N)
            
            for a in powerfuls_list_local[:end_idx]:
                b = N - a
                if b in powerfuls_set_local:
                    if a <= b:
                        pairs.append([int(a), int(b)])
            
            # Calculate score
            # The score is the number of pairs.
            if len(pairs) > best_score:
                best_score = len(pairs)
                best_N = N
                best_pairs = pairs
                
            # Time check
            if time.time() > deadline - 5:
                break
        
        if best_N is not None:
            return {'N': best_N, 'pairs': best_pairs}
        
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    except MemoryError:
        # Fallback to smaller L
        L = 10_000_000
        powerfuls = _generate_powerfuls(L)
        powerfuls_list = sorted(powerfuls)
        powerfuls_set = set(powerfuls_list)
        
        # Brute force for small L
        count_dict = {}
        pair_dict = {}
        for i in range(len(powerfuls_list)):
            a = powerfuls_list[i]
            for j in range(i, len(powerfuls_list)):
                b = powerfuls_list[j]
                s = a + b
                if s > 2 * L: break
                if s not in count_dict:
                    count_dict[s] = 0
                    pair_dict[s] = []
                count_dict[s] += 1
                pair_dict[s].append([a, b])
        
        if count_dict:
            best_N = max(count_dict, key=count_dict.get)
            return {'N': best_N, 'pairs': pair_dict[best_N]}
        return {'N': 2, 'pairs': [[1, 1]]}
        
    except Exception:
        return {'N': 2, 'pairs': [[1, 1]]}

# EVOLVE-BLOCK-END
