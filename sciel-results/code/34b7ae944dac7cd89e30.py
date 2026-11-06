import time
import numpy as np
from math import isqrt
from collections import defaultdict

def generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    pows = set([1])
    # Powerful numbers are of the form a^2 * b^3 where b is square-free.
    # We iterate over square-free b and all a.
    max_b = int(limit ** (1/3)) + 1
    
    # Precompute square-free numbers up to max_b
    def is_square_free(n):
        if n <= 1:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    for b in range(1, max_b + 1):
        if not is_square_free(b):
            continue
        
        b_cubed = b ** 3
        if b_cubed > limit:
            break
            
        max_a = isqrt(limit // b_cubed)
        for a in range(1, max_a + 1):
            n = a * a * b_cubed
            if n <= limit:
                pows.add(n)
            else:
                break
                
    return sorted(pows)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # We want to maximize K (number of powerful numbers) to maximize the chance of finding a high-frequency sum.
    # The sum of the two largest powerful numbers will be the max N we can represent with pairs within the list.
    # If limit = 10^7, K ~ 8000. Sum max ~ 2*10^7.
    # If limit = 10^8, K ~ 25000. Sum max ~ 2*10^8.
    # If limit = 10^9, K ~ 80000. Sum max ~ 2*10^9.
    # If limit = 10^10, K ~ 250000. Sum max ~ 2*10^10.
    
    # NumPy histogram or convolution is O(K log K) or O(N log N) where N is max sum.
    # Using np.bincount on sums is O(K^2) if we generate all pairs, which is bad.
    # But we can use the fact that we just want the frequency of sums.
    # Wait, frequency of sums of two elements from a set A.
    # This is equivalent to the coefficient of x^k in (sum_{a in A} x^a)^2.
    # We can compute this via FFT/Convolution.
    # However, the range of sums is up to 2*limit.
    # For limit=10^8, range is 2*10^8. FFT of size 2^28 ~ 268M. This might be too large for memory/time in Python/NumPy within 130s?
    # Let's check memory. 268M complex128 is ~ 4GB. Too much.
    # 2^27 ~ 134M. ~ 2GB. Still risky.
    # 2^26 ~ 67M. ~ 1GB.
    
    # Alternative:
    # If we pick limit = 10^7, max sum is 2*10^7.
    # Array size 2*10^7. Convolution is fast.
    # K ~ 8000.
    # Current program does O(K^2) = 64M operations in Python loops. This is slow (~10-30s).
    # NumPy vectorized pair generation is also O(K^2) but faster.
    # But Convolution is O(N log N). For N=2*10^7, log N ~ 25. 2*10^7 * 25 = 5*10^8 operations.
    # FFT in NumPy is highly optimized C. This should be much faster than 64M Python loop iterations.
    
    # Let's try limit = 10^8.
    # Max sum = 2*10^8.
    # Next power of 2 is 2^28 = 268,435,456.
    # Memory: 268M * 8 bytes (complex) * 2 (real/imag or just complex) = ~ 4.3 GB.
    # This might exceed memory limits or be slow to allocate.
    
    # Let's try limit = 5*10^7.
    # Max sum = 10^8.
    # Next power of 2 is 2^27 = 134,217,728.
    # Memory: 134M * 16 bytes (complex128) = ~ 2.1 GB.
    # This is borderline but likely feasible on standard evaluation hardware.
    # K for 5*10^7 is approx 5*8000 = 40000? No, K scales as L^(2/3).
    # K(10^7) ~ 8000.
    # K(5*10^7) = 8000 * (5)^(2/3) ~ 8000 * 2.92 ~ 23000.
    # K(10^8) = 8000 * (10)^(2/3) ~ 8000 * 4.64 ~ 37000.
    
    # Let's start with limit = 10^8 and see if it fits. If it crashes or times out on allocation, we fallback.
    # Actually, let's be safe and use limit = 10^7 first to guarantee correctness, then try to expand if time permits?
    # No, the score depends on the count. A larger limit yields a larger K, which yields a higher expected max frequency.
    # Frequency of sums for random-ish numbers of size K in range M is approx K^2 / M.
    # Here M ~ 2L. K ~ L^(2/3).
    # Frequency ~ L^(4/3) / L = L^(1/3).
    # So increasing L from 10^7 to 10^8 increases expected max frequency by 10^(1/3) ~ 2.15.
    # Current score 4.06. Expected new score ~ 8.7.
    
    # Let's try L = 10^8.
    limit = 10**8
    
    powerful_list = generate_powerful(limit)
    K = len(powerful_list)
    
    if time.time() > deadline - 20:
        # Fallback to smaller limit if generation took too long
        limit = 10**7
        powerful_list = generate_powerful(limit)
        K = len(powerful_list)

    # Convert to numpy array
    pows = np.array(powerful_list, dtype=np.int64)
    
    # We want to find the N with the maximum number of pairs (a, b) such that a+b=N, a,b in pows.
    # This is the coefficient of the convolution of the indicator function of pows with itself.
    
    max_sum = 2 * pows[-1]
    
    # Determine FFT size
    # We need size >= max_sum + 1
    # Next power of 2
    fft_size = 1
    while fft_size < max_sum + 1:
        fft_size *= 2
        
    # Check memory and time
    # If fft_size is too large, we might fail.
    # 2^27 is ~ 134M. 2^28 is ~ 268M.
    # Let's cap fft_size to avoid OOM. If max_sum > 10^8, we might have issues.
    # For L=10^8, max_sum = 2*10^8. fft_size = 2^28.
    # 268M complex128 is 4.3 GB.
    # If we run out of memory, we should catch it.
    
    try:
        # Create indicator vector
        indicator = np.zeros(fft_size, dtype=np.complex128)
        indicator[pows] = 1.0
        
        # FFT
        t_start_fft = time.time()
        spectrum = np.fft.fft(indicator)
        
        # Square spectrum (convolution in frequency domain)
        # Note: This computes circular convolution. Since we padded to power of 2 >= max_sum+1,
        # and max_sum < fft_size, there is no aliasing for the first max_sum+1 coefficients.
        spectrum_squared = spectrum * spectrum
        
        # Inverse FFT
        conv = np.fft.ifft(spectrum_squared).real
        
        # The result should be integers (counts).
        # Due to floating point errors, they might be 1.9999999 or 2.0000001.
        # We round them.
        counts = np.round(conv[:max_sum + 1]).astype(np.int64)
        
        # Find the index with the maximum count
        # We only care about N >= 2 (since a,b >= 1, min sum is 2)
        # Also, we need to ensure that the pair (a, b) with a <= b is counted correctly.
        # The convolution counts ordered pairs (a, b).
        # If a != b, (a, b) and (b, a) are distinct in convolution, so count is 2 * (number of unordered pairs with a!=b).
        # If a == b, (a, a) is counted once.
        # The task asks for distinct pairs a <= b.
        # So for a given N:
        # Let C be the convolution count (ordered pairs).
        # Let D be the number of pairs with a == b (i.e., N is even and N/2 is powerful).
        # Then number of unordered pairs = (C - D) / 2 + D = (C + D) / 2.
        # Wait.
        # Ordered pairs: sum_{a,b in P, a+b=N} 1.
        # Unordered pairs a<=b: 
        # If a < b, counted twice in ordered.
        # If a = b, counted once in ordered.
        # So Ordered = 2 * (# pairs a<b) + 1 * (# pairs a=b).
        # Unordered = # pairs a<b + # pairs a=b.
        # Unordered = (Ordered - # pairs a=b) / 2 + # pairs a=b = (Ordered + # pairs a=b) / 2.
        
        # We need to calculate D for the best N.
        # First, find the N that maximizes Unordered count.
        # Unordered(N) = (Conv(N) + D(N)) / 2.
        # D(N) is 1 if N is even and N/2 is in powerful_set, else 0.
        
        # We can compute D for all N efficiently?
        # D(N) = 1 if N/2 in P.
        # Let's create an array D of size max_sum+1.
        # D[2*p] = 1 for p in P.
        
        D = np.zeros(max_sum + 1, dtype=np.int64)
        # 2*pows might exceed max_sum? No, max_sum = 2*max(pows).
        # So 2*pows is within bounds.
        D[2 * pows] = 1
        
        unordered_counts = (counts + D) // 2
        
        # Find the max
        # We should ignore N < 2?
        # pows contains 1. Min sum 1+1=2.
        # So indices 0 and 1 are 0.
        
        max_idx = np.argmax(unordered_counts)
        best_N = max_idx
        best_count = unordered_counts[max_idx]
        
        # If the time is running out, we still need to return valid pairs.
        # We have the count. Now we need to list the pairs.
        # This requires iterating through pows and checking if best_N - a is in pows.
        # This is O(K).
        
        powerful_set = set(powerful_list)
        pairs = []
        for a in powerful_list:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in powerful_set and b >= a:
                pairs.append([a, b])
        
        # Verify count
        # If len(pairs) != best_count, something is wrong (e.g. floating point error in FFT rounding).
        # If it's off by 1 or 2, we can trust the list of pairs.
        # The score is the number of pairs in the list.
        
        return {'N': int(best_N), 'pairs': pairs}
        
    except MemoryError:
        # Fallback to smaller limit if FFT is too big
        limit = 10**7
        powerful_list = generate_powerful(limit)
        pows = np.array(powerful_list, dtype=np.int64)
        max_sum = 2 * pows[-1]
        fft_size = 1
        while fft_size < max_sum + 1:
            fft_size *= 2
        
        indicator = np.zeros(fft_size, dtype=np.complex128)
        indicator[pows] = 1.0
        spectrum = np.fft.fft(indicator)
        spectrum_squared = spectrum * spectrum
        conv = np.fft.ifft(spectrum_squared).real
        counts = np.round(conv[:max_sum + 1]).astype(np.int64)
        
        D = np.zeros(max_sum + 1, dtype=np.int64)
        D[2 * pows] = 1
        unordered_counts = (counts + D) // 2
        
        max_idx = np.argmax(unordered_counts)
        best_N = max_idx
        
        powerful_set = set(powerful_list)
        pairs = []
        for a in powerful_list:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in powerful_set and b >= a:
                pairs.append([a, b])
                
        return {'N': int(best_N), 'pairs': pairs}
    except Exception:
        # Fallback to pure python if NumPy fails
        limit = 10**7
        powerful_list = generate_powerful(limit)
        powerful_set = set(powerful_list)
        rep_count = defaultdict(int)
        num_powerful = len(powerful_list)
        
        for i in range(num_powerful):
            a = powerful_list[i]
            for j in range(i, num_powerful):
                b = powerful_list[j]
                N = a + b
                rep_count[N] += 1
            if time.time() > deadline - 5:
                break
        
        if not rep_count:
            return {'N': 2, 'pairs': [[1, 1]]}
            
        best_N = max(rep_count.keys(), key=lambda n: rep_count[n])
        pairs = []
        for a in powerful_list:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in powerful_set and b >= a:
                pairs.append([a, b])
                
        return {'N': best_N, 'pairs': pairs}
