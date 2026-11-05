import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    timeout = 145  # safety margin before 150s limit
    
    # Strategy: Generate powerful numbers up to M, then use a smarter search
    # for N with many representations. Instead of dense array for all sums,
    # we'll generate powerful numbers up to M, then for each powerful number a,
    # we check how many powerful numbers b exist such that a+b = N for various N.
    # 
    # Better approach: Generate powerful numbers, sort them. For each pair of 
    # "hot" regions, find N values that maximize the count.
    #
    # Even better: Use the fact that powerful numbers are sparse. Generate them
    # up to M, then for each powerful number a, compute N = a + b for all b,
    # and count occurrences. But this is O(K^2) which is too slow for large K.
    #
    # Alternative: Use FFT on a sparse representation. But the sum range is 2*M.
    # For M = 50M, that's 100M entries in the FFT, which is feasible.
    #
    # Let's try: Generate powerful numbers up to M, create a boolean array of size 2*M,
    # then use FFT to compute the convolution (which gives pair counts).
    
    # Choose M based on what's feasible
    # Powerful numbers up to M: count is roughly M^(1/3) * (log M)^(1/3) * constant
    # For M = 10^8, count ~ 5000 * 5 * 0.7 ~ 17500 (rough)
    # Actually more like: sum over y of sqrt(M/y^3) = M^(1/2) * sum(y^(-3/2)) ~ M^(1/2) * 2.6
    # For M = 10^8: ~ 10^4 * 2.6 = 26000
    # For M = 10^9: ~ 31623 * 2.6 = 82000
    # For M = 10^10: ~ 10^5 * 2.6 = 260000
    
    # Let's use M = 200_000_000 to get a good density
    M = 200_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            if p <= M:
                powerful.add(p)
    
    powerful_list = sorted(powerful)
    n_list = len(powerful_list)
    
    if n_list < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Use FFT to compute pair counts
    # Create boolean array of size 2*M
    size = 2 * M
    # We need array of size at least 2*M for the convolution
    # FFT size should be next power of 2 >= 2*M
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    # Check if we have enough time
    if time.time() - start_time > timeout - 30:
        # Fallback: just use a smaller approach
        M_small = 50_000_000
        powerful_small = set()
        max_y_small = int(M_small ** (1/3)) + 1
        for y in range(1, max_y_small + 1):
            y3 = y ** 3
            if y3 > M_small:
                break
            max_x = isqrt(M_small // y3)
            for x in range(1, max_x + 1):
                p = x * x * y3
                if p <= M_small:
                    powerful_small.add(p)
        powerful_list_small = sorted(powerful_small)
        powerful_set_small = set(powerful_list_small)
        
        # For smaller set, do O(K^2) with early termination
        best_N = 2
        best_count = 1
        best_pairs = [[1, 1]]
        
        counts = {}
        for i in range(len(powerful_list_small)):
            if time.time() - start_time > timeout - 5:
                break
            a = powerful_list_small[i]
            for j in range(i, len(powerful_list_small)):
                b = powerful_list_small[j]
                s = a + b
                if s > 2 * M_small:
                    break
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > best_count:
                    best_count = counts[s]
                    best_N = s
                    # Reconstruct pairs
                    best_pairs = []
                    for aa in powerful_list_small:
                        if aa > best_N // 2:
                            break
                        bb = best_N - aa
                        if bb in powerful_set_small:
                            best_pairs.append([aa, bb])
        
        return {'N': best_N, 'pairs': best_pairs}
    
    # Create boolean array
    arr = np.zeros(fft_size, dtype=np.float64)
    for p in powerful_list:
        arr[p] = 1.0
    
    # FFT
    fft_arr = np.fft.fft(arr)
    fft_arr *= fft_arr
    result = np.fft.ifft(fft_arr).real
    
    # The result[i] gives the number of pairs (a,b) with a+b=i
    # We want a <= b, so we need to handle the diagonal
    # result[i] = count of (a,b) with a+b=i, a,b in powerful
    # For a < b: counted once in result
    # For a = b: counted once
    # We want pairs with a <= b, so:
    # number of pairs = (result[i] + (1 if i is even and i/2 is powerful else 0)) / 2
    # Actually: result[i] = sum over all a in powerful of [i-a in powerful]
    # = count of ordered pairs (a,b) with a+b=i
    # Number of unordered pairs with a<=b = (result[i] + diag[i]) / 2
    # where diag[i] = 1 if i is even and i/2 is powerful, else 0
    
    # Find the best N
    # We need N >= 2 and N <= 2*M
    # Compute adjusted counts
    # For efficiency, just find argmax of result in range [2, 2*M]
    
    valid_range = result[2:size+1]
    best_idx = np.argmax(valid_range)
    best_N = int(best_idx + 2)
    best_raw_count = int(round(result[best_N]))
    
    # Compute actual number of pairs with a <= b
    powerful_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
    
    best_count = len(pairs)
    
    # Check if we should try nearby N values with higher actual counts
    # The FFT gives approximate counts due to floating point
    # Let's verify a few top candidates
    top_indices = np.argsort(valid_range)[::-1][:20]
    best_actual = best_count
    best_N_final = best_N
    best_pairs_final = pairs
    
    for idx in top_indices:
        if time.time() - start_time > timeout - 10:
            break
        N_candidate = int(idx + 2)
        if N_candidate < 2 or N_candidate > 2 * M:
            continue
        # Count actual pairs
        cnt = 0
        for a in powerful_list:
            if a > N_candidate // 2:
                break
            b = N_candidate - a
            if b in powerful_set:
                cnt += 1
        if cnt > best_actual:
            best_actual = cnt
            best_N_final = N_candidate
            best_pairs_final = []
            for a in powerful_list:
                if a > best_N_final // 2:
                    break
                b = best_N_final - a
                if b in powerful_set:
                    best_pairs_final.append([a, b])
    
    if best_actual < 1:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N_final, 'pairs': best_pairs_final}
