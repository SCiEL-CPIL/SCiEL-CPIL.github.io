import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form u^2 * v^3 where v is square-free.
    However, the standard generation u^2 * v^3 (without square-free constraint on v)
    generates each powerful number multiple times (e.g., 36 = 6^2*1^3 = 1^2*2^3*... no, 36=2^2*3^2. 
    36 = 6^2 * 1^3. Also 36 = (2*3)^2. 
    Is 36 = u^2 v^3 with v square-free?
    36 = 2^2 * 3^2.
    If v=1, u=6. 6^2*1 = 36.
    If v=2 (sf), u^2 * 8 = 36 -> u^2=4.5 no.
    If v=3 (sf), u^2 * 27 = 36 -> no.
    So 36 is generated once by v=1, u=6.
    
    What about 72 = 2^3 * 3^2?
    v must contain the primes with odd exponents in the factorization?
    Actually, any powerful number n can be written uniquely as u^2 v^3 where v is square-free.
    Proof: Let n = prod p_i^{e_i}.
    If e_i is even, p_i goes into u.
    If e_i is odd, p_i goes into v (to the power 1, since v is square-free) and p_i^{e_i-1} into u (which is even power).
    Since e_i >= 2 for powerful numbers, if e_i is odd, e_i >= 3, so e_i-1 >= 2.
    So this representation is unique if v is square-free.
    
    The previous code checked if b (v) was square-free.
    The inspiration code just iterates all v and adds u^2 v^3.
    This generates duplicates in the set, but `set` handles that.
    The complexity is sum_{v=1}^{L^{1/3}} sqrt(L/v^3) = sqrt(L) * sum v^{-1.5} ~ 2 sqrt(L).
    For L=10^8, this is 20,000 operations. Very fast.
    For L=10^9, 20,000.
    For L=10^12, 200,000.
    
    The previous code with square-free check was also fast but slightly more complex.
    The simpler code (inspiration) is likely faster due to less overhead per v, even with duplicates.
    """
    powerful = set()
    max_v = int(limit ** (1.0 / 3.0)) + 2
    
    for v in range(1, max_v + 1):
        v3 = v * v * v
        if v3 > limit:
            break
        max_u = isqrt(limit // v3)
        for u in range(1, max_u + 1):
            val = u * u * v3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s safety margin
    deadline = start_time + 140.0

    # We want to maximize the number of pairs.
    # The number of pairs for N is roughly proportional to the density of powerful numbers near N/2.
    # Powerful numbers have density ~ C / sqrt(x) * log(x)^{-1/3}? No, density is ~ x^{-1/2}.
    # So larger N (up to 10^18) should have more representations if we can generate enough powerful numbers.
    # However, we are limited by memory for the FFT/convolution.
    
    # Strategy:
    # 1. Choose a limit M such that the FFT of size 2*M+1 fits in memory.
    # 2. Generate powerful numbers up to M.
    # 3. Use FFT to find N with max representations.
    # 4. Verify and extract pairs.
    
    # Memory estimate for FFT:
    # np.fft.rfft on array of size S uses ~ S * 8 bytes (complex128) for intermediate?
    # Actually, numpy FFT is out-of-place usually? Or in-place?
    # Let's assume we need 2*S * 8 bytes = 16*S bytes for safety (input + output complex).
    # If S = 10^8, 1.6 GB. This might be tight.
    # If S = 5*10^7, 800 MB. Safer.
    # Let's try S = 2*M+1.
    # Target M = 20,000,000 -> S = 40,000,001.
    # Memory ~ 40M * 16 = 640 MB. Plus Python overhead. Should be safe.
    
    # Can we go higher?
    # M = 50,000,000 -> S = 100,000,001.
    # Memory ~ 1.6 GB. Risky if other stuff is loaded.
    # Let's try M = 30,000,000 first.
    
    limits_to_try = [50_000_000, 30_000_000, 10_000_000]
    
    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    for M in limits_to_try:
        try:
            if time.time() > deadline - 30:
                break
                
            # 1. Generate powerful numbers
            gen_start = time.time()
            powerful_list = _generate_powerful_numbers(M)
            gen_time = time.time() - gen_time if False else time.time() - gen_start
            
            if time.time() > deadline - 20:
                continue
                
            K = len(powerful_list)
            if K == 0:
                continue
            
            # 2. FFT Convolution
            # We want to count pairs (a,b) such that a+b=N, a,b in powerful_list.
            # This is the auto-correlation of the indicator function of powerful numbers.
            # Let A be an array of size S where A[x] = 1 if x in powerful_list, 0 else.
            # Then FFT(A) * FFT(A) -> IFFT -> C.
            # C[n] = sum A[i] A[n-i] = number of pairs (i, n-i) in powerful_list.
            # This counts ordered pairs.
            # We want unordered pairs a <= b.
            # If a != b, (a,b) and (b,a) are both counted. So 2 * count.
            # If a == b, (a,a) is counted once.
            # Let C_total = C[n].
            # Let D = # pairs with a < b.
            # Let E = # pairs with a = b (1 if n/2 in powerful_list, 0 else).
            # C_total = 2D + E.
            # We want Pairs = D + E = (C_total + E) / 2.
            
            # Size S must be at least 2*M + 1 to avoid circular convolution effects.
            # Max sum is 2*M. So indices 0..2*M.
            S = 2 * M + 1
            
            # Create indicator array
            # Use float32? FFT usually works better with float64 for precision.
            # But float32 saves memory.
            # Max value in C is K. K ~ 2*sqrt(M). For M=50M, K~14000.
            # float32 precision is ~7 digits. 14000 is fine.
            # However, IFFT can introduce small errors. We will round to nearest int.
            
            arr = np.zeros(S, dtype=np.float64)
            # powerful_list contains ints.
            arr[powerful_list] = 1.0
            
            # Perform FFT
            fft_arr = np.fft.rfft(arr)
            conv = np.fft.irfft(fft_arr * fft_arr, n=S)
            
            # conv[i] is the count of ordered pairs.
            
            # 3. Find best N
            # We want to maximize (conv[i] + E_i) / 2.
            # E_i is 0 or 1.
            # So maximizing conv[i] is a very good proxy.
            # Let's find top candidates.
            
            # Get top 100 indices by conv value
            top_k = 100
            if S > top_k:
                # argpartition is O(N)
                top_indices = np.argpartition(conv, -top_k)[-top_k:]
            else:
                top_indices = np.arange(S)
                
            # Sort these top indices by conv value descending
            top_indices = top_indices[np.argsort(conv[top_indices])[::-1]]
            
            best_N = 2
            best_count = 1
            best_pairs = [[1, 1]]
            
            powerful_set = set(powerful_list)
            
            # Verify top candidates
            for idx in top_indices:
                if time.time() > deadline - 5:
                    break
                    
                N_cand = int(idx)
                
                # Calculate exact number of pairs a <= b
                exact_pairs = []
                count = 0
                # Iterate a in powerful_list such that a <= N_cand // 2
                # powerful_list is sorted.
                # We can break early when a > N_cand // 2
                
                half_N = N_cand // 2
                
                for a in powerful_list:
                    if a > half_N:
                        break
                    b = N_cand - a
                    if b in powerful_set:
                        count += 1
                        exact_pairs.append([a, b])
                        
                if count > best_count:
                    best_count = count
                    best_N = N_cand
                    best_pairs = exact_pairs
                    
                # Early termination check:
                # If conv[idx] < 2 * best_count, then even if E=1,
                # (conv[idx] + 1) / 2 <= (2*best_count - 1 + 1) / 2 = best_count.
                # So it cannot strictly exceed best_count.
                # Since top_indices is sorted by conv descending, we can break.
                if conv[idx] < 2 * best_count:
                    break
            
            # Update best result
            if best_count > len(best_result['pairs']):
                best_result = {'N': best_N, 'pairs': best_pairs}
                
            # If we succeeded with a large M, it's likely the best.
            # Smaller M will have fewer powerful numbers, hence lower max count.
            # So we can return immediately if we tried a large M.
            # But just in case, let's return the best found so far.
            # Since we iterate M descending, the first successful large M is best.
            return best_result

        except MemoryError:
            continue
        except Exception as e:
            # Fallback to next smaller M
            continue

    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
