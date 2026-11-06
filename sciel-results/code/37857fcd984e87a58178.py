import time
import math
import numpy as np

def construct():
    start_time = time.time()
    # Set a deadline 10 seconds before the 120s limit
    deadline = start_time + 110.0

    best_N = 2
    best_pairs = [[1, 1]]
    best_score = 1

    # Limits to try in increasing order
    # We scale up if we have time and if the previous result wasn't already very good
    # Note: K ~ sqrt(L) + L^(1/3).
    # L=1e7 -> K~3500, K^2=1.2e7 (fast, small mem)
    # L=5e7 -> K~7500, K^2=5.6e7 (fast, ~450MB mem)
    # L=2e8 -> K~14000, K^2=2e8 (slow mem ~1.6GB, risky) -> Use FFT
    
    limits = [10_000_000, 30_000_000, 80_000_000, 200_000_000]
    
    for limit in limits:
        if time.time() > deadline - 15.0:
            break
            
        # 1. Generate powerful numbers up to limit
        pows_set = {1}
        max_i = math.isqrt(limit)
        
        # Pre-compute cubes
        max_j_global = int(round(limit ** (1/3))) + 2
        cubes = [j ** 3 for j in range(1, max_j_global + 1)]
        
        # Generate powerful numbers
        # val = i^2 * j^3
        for i in range(1, max_i + 1):
            sq = i * i
            if sq > limit:
                break
            max_j = int(round((limit // sq) ** (1/3)))
            if max_j > max_j_global:
                max_j = max_j_global
            
            for j in range(1, max_j + 1):
                val = sq * cubes[j-1]
                if val <= limit:
                    pows_set.add(val)
                else:
                    break
            
            if time.time() > deadline - 10.0:
                break
        
        P_list = sorted(pows_set)
        K = len(P_list)
        
        if K == 0:
            continue

        P = np.array(P_list, dtype=np.int64)
        
        # 2. Count representations
        # Strategy:
        # If K is small enough, use direct NumPy sum and bincount.
        # If K is large, use FFT.
        
        try:
            if K < 12000:
                # Direct method
                # Memory: K^2 * 8 bytes. 12000^2 * 8 = 115 MB. Safe.
                sums = P[:, None] + P[None, :]
                max_sum = 2 * limit
                # bincount requires int array. sums are int64.
                # Flatten to 1D
                flat_sums = sums.ravel()
                counts = np.bincount(flat_sums, minlength=max_sum + 1)
                
                # Find top candidates
                # We want max (counts[s] + diag) // 2
                # approx max counts[s]
                # Get top 20 indices
                c_slice = counts[2:]
                # Handle case where c_slice is small
                if len(c_slice) < 20:
                    top_indices_local = np.argsort(c_slice)[-len(c_slice):]
                else:
                    top_indices_local = np.argpartition(c_slice, -20)[-20:]
                top_indices = top_indices_local + 2
                
            else:
                # FFT Method
                # Create indicator vector
                size = 2 * limit + 1
                # Next power of 2 for FFT efficiency?
                # scipy.fft.rfft is available. 
                # Let's use numpy.fft.rfft for simplicity, though scipy is faster.
                # But numpy is standard.
                
                # Allocate vector
                # If size is too big for memory (e.g. > 10^9), this will fail.
                # limit=2e8 -> size=4e8. float64 = 3.2GB. Might be too much.
                # Let's check available memory? No, just try.
                # If MemoryError, we skip this limit.
                
                vec = np.zeros(size, dtype=np.float64)
                vec[P] = 1.0
                
                # FFT
                # rfft returns complex values of length size//2 + 1
                # Convolution is inverse FFT of (FFT(vec) * FFT(vec))
                # We only need real part of the inverse.
                
                # Note: numpy.fft.rfft is O(N log N).
                # Inverse is irfft.
                
                fft_vec = np.fft.rfft(vec)
                conv = np.fft.irfft(fft_vec * fft_vec, n=size)
                
                # conv[i] is the count of pairs (a,b) such that a+b=i
                # Due to floating point errors, counts might be 4.9999 or 5.0001
                # We round them.
                counts = np.rint(conv).astype(np.int64)
                
                # Find top candidates
                c_slice = counts[2:]
                if len(c_slice) < 20:
                    top_indices_local = np.argsort(c_slice)[-len(c_slice):]
                else:
                    top_indices_local = np.argpartition(c_slice, -20)[-20:]
                top_indices = top_indices_local + 2

        except MemoryError:
            # If FFT or Direct method fails due to memory, skip to next (larger) limit?
            # No, larger limit will be worse. 
            # If we are here, it means K was large and FFT failed, or K was small and direct failed.
            # If K < 12000 and direct failed, memory is very tight.
            # If K >= 12000 and FFT failed, memory is very tight.
            # In either case, try the next limit? No, next limit is larger.
            # Break out and return best so far.
            break
        except Exception:
            # If any error, skip this limit
            continue

        # 3. Verify top candidates
        # We have top_indices which are sums s.
        # We need to count distinct pairs a<=b such that a+b=s and a,b in pows_set.
        
        for s in top_indices:
            s_val = int(s)
            if s_val > 2 * limit or s_val < 2:
                continue
            
            # Count pairs
            pairs = []
            half = s_val / 2.0
            
            # Iterate through P_list
            # P_list is sorted.
            for a in P_list:
                if a > half:
                    break
                b = s_val - a
                if b in pows_set:
                    pairs.append([a, b])
            
            score = len(pairs)
            if score > best_score:
                best_score = score
                best_N = s_val
                best_pairs = pairs
            
            if time.time() > deadline - 5.0:
                break
        
        # If we found a very high score, we might stop early?
        # The score is the count. Max possible is K/2.
        # If best_score is close to K/2, it's likely optimal for this limit.
        # But we don't know the global max. 
        # We continue to next limit if time permits, hoping for a larger K and thus higher score.
        
    return {'N': best_N, 'pairs': best_pairs}
