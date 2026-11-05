import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    # Hard deadline with safety margin (120s limit, aim for 100s)
    deadline = start_time + 100.0

    # We want to find N <= 10^18 with many representations a+b=N where a,b are powerful.
    # The seed program searched up to 20,000. 
    # Powerful numbers are sparse. Density is roughly x^{1/2} / log(x)^{1/2} ? No, count is ~ c * x^{1/2}.
    # Actually number of powerful numbers <= X is ~ C * X^{1/2} / log(X)^{1/2} is wrong.
    # It is ~ C * X^{1/2}. Wait, sum_{k^2 <= X} ... no.
    # Powerful numbers n = a^2 b^3.
    # Count is O(X^{1/2}).
    # If we search up to M, we have ~ sqrt(M) powerful numbers.
    # Pairs are ~ M / log M ? No.
    # Let's just increase M significantly.
    # M = 10^7 allows FFT of size 2*10^7 which is fast in numpy.
    # M = 10^8 might be too slow for FFT in Python without scipy.fft or careful chunking, but 10^7 is safe.
    
    M = 10_000_000  # 10 million
    
    # 1. Generate powerful numbers up to M
    # Powerful number n has all prime factors with exponent >= 2.
    # n = a^2 * b^3 where b is square-free? No, that's a specific form.
    # Simpler: iterate over squares and cubes.
    # A number is powerful if for every prime p | n, p^2 | n.
    # Efficient generation:
    # n = x^2 * y^3 is not the only form? 
    # Actually, any powerful number can be written as a^2 * b^3 where b is square-free?
    # Let n = prod p_i^{e_i}. e_i >= 2.
    # Let e_i = 2*k_i + r_i where r_i in {0,1}.
    # Then n = (prod p_i^{k_i})^2 * (prod p_i^{r_i})^3 ? No.
    # If e_i is even, it's in the square part. If odd, e_i >= 3, so e_i-3 is even.
    # So n = (prod p_i^{(e_i - r_i)/2})^2 * (prod p_i^{r_i})^3 where r_i is e_i mod 2?
    # If e_i is even, r_i=0, term is 1.
    # If e_i is odd, r_i=1, term is p_i.
    # So n = A^2 * B^3 where B is square-free.
    # Yes.
    
    powerful = set()
    
    # Max B such that B^3 <= M is M^(1/3) ~ 215
    # Max A such that A^2 * 1 <= M is M^(1/2) ~ 3162
    
    max_b = int(M ** (1/3)) + 1
    max_a = int(M ** 0.5) + 1
    
    # We need to generate all n = a^2 * b^3 <= M
    # b must be square-free?
    # If b is not square-free, say b = k^2 * c, then b^3 = k^6 * c^3.
    # Then a^2 * b^3 = (a * k^3)^2 * c^3.
    # So we can just iterate b up to M^(1/3) and a up to M^(1/2)/b^(3/2).
    # But wait, if we just generate all a^2 * b^3, do we get duplicates?
    # Yes, but we use a set.
    # Do we miss any powerful numbers?
    # Any powerful number n has a unique representation n = A^2 * B^3 where B is square-free.
    # If we iterate all b, we will generate the number multiple times if b is not square-free, 
    # but we will definitely generate it at least once with the square-free part.
    # Actually, if we just iterate all b from 1 to M^(1/3), we cover all cases.
    
    # Optimization: Only iterate b that are square-free?
    # It's easier to just iterate all b and use a set.
    # M^(1/3) is ~ 215. M^(1/2) is ~ 3162.
    # Total iterations ~ sum_{b=1}^{215} (3162 / b^1.5) ~ 3162 * sum 1/b^1.5 ~ 3162 * 2 ~ 6000.
    # This is very fast.
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > M:
            break
        # a^2 <= M / b3
        max_a_for_b = int((M // b3) ** 0.5)
        for a in range(1, max_a_for_b + 1):
            n = a * a * b3
            if n <= M:
                powerful.add(n)
            else:
                break
                
    if time.time() > deadline - 5:
         # Fallback to small search if generation was unexpectedly slow (unlikely)
         M = 20000
         powerful = {a * a * b ** 3 for b in range(1, 28) for a in range(1, isqrt(M // b ** 3) + 1)}

    powerful_list = sorted(powerful)
    
    # 2. FFT Convolution to count pairs
    # We want c[n] = sum_{a+b=n} 1[a in P] 1[b in P]
    # Create array v where v[i] = 1 if i in P else 0.
    # Convolve v with v.
    # c[k] = sum_{i=0}^k v[i] v[k-i]
    # This counts ordered pairs (a,b).
    # Since a,b are integers, and we want a <= b, 
    # if a != b, (a,b) and (b,a) are distinct in convolution but same pair in task.
    # If a == b, it's 1 pair.
    # Task asks for distinct pairs a <= b.
    # So count(n) = (conv[n] + (1 if n is even and n/2 in P else 0)) / 2 ?
    # Let's check.
    # Conv[n] = sum_{a} v[a] v[n-a].
    # If a != n-a, term is 1 for (a, n-a) and 1 for (n-a, a). Total 2.
    # If a == n-a (i.e. n even, a=n/2), term is v[a]^2 = 1. Total 1.
    # So number of pairs a<=b is (Conv[n] + diag[n]) / 2, where diag[n]=1 if n/2 in P else 0.
    
    # We only care about n up to M? 
    # Wait, N can be up to 10^18. 
    # The seed program searched for N up to M.
    # Can we find a larger N with more representations?
    # The density of powerful numbers decreases as x increases.
    # Number of reps R(N) is roughly proportional to density of P near N/2.
    # Since density decreases, smaller N likely has more reps.
    # So searching N up to M is a good heuristic.
    # However, M=10^7 is much larger than 20,000.
    
    # FFT size
    # We need convolution up to index M (since a,b <= M, a+b <= 2M? No, a,b are powerful <= M, so a+b <= 2M).
    # But we only want N <= M? Or can N be up to 2M?
    # The task says N <= 10^18.
    # If we generate P up to M, we can form sums up to 2M.
    # Should we search N up to 2M?
    # Density is lower for larger N. So N in [M, 2M] will have fewer reps than N in [M/2, M].
    # So searching N up to M is sufficient and faster.
    
    # Create vector
    v = np.zeros(M + 1, dtype=np.float64)
    for p in powerful_list:
        v[p] = 1.0
        
    # FFT
    # Size must be at least 2*M to avoid circular convolution aliasing for sums up to 2M.
    # But we only care about sums up to M.
    # Linear convolution of length M+1 and M+1 has length 2M+1.
    # We need FFT size >= 2M+1.
    # Next power of 2.
    n_fft = 1
    while n_fft < 2 * M + 1:
        n_fft *= 2
        
    if time.time() > deadline - 20:
        # If we are running out of time, maybe reduce M?
        # But we are already in FFT. FFT is fast.
        pass
        
    V = np.fft.rfft(v, n=n_fft)
    V2 = V * V
    conv = np.fft.irfft(V2, n=n_fft)
    
    # conv[k] is the count of ordered pairs (a,b) such that a+b=k.
    # We want to find N in [2, M] that maximizes number of pairs a<=b.
    # pairs_count[N] = (conv[N] + (1 if N % 2 == 0 and int(N/2) in powerful else 0)) / 2
    
    # Vectorized calculation for N in 2..M
    Ns = np.arange(2, M + 1)
    conv_vals = conv[2:M+1]
    
    # Check diagonal
    # N is even -> N/2 is integer.
    # Check if N/2 is in powerful.
    # We can create a boolean array for powerful.
    # P_bool = np.zeros(M+1, dtype=bool)
    # P_bool[powerful_list] = True
    
    # But we already have v. v[i] > 0 iff i in P.
    # So diag is 1 if N is even and v[N//2] == 1.
    
    # Let's compute scores
    # scores = (conv_vals + diag) / 2
    
    # To be safe and fast, let's just use conv_vals as a proxy?
    # No, the diagonal correction is small (0 or 1), while conv_vals can be large.
    # But for small N, conv_vals might be small.
    # Let's do it properly.
    
    # Create mask for even Ns
    even_mask = (Ns % 2 == 0)
    
    # Indices for N/2
    half_Ns = Ns // 2
    
    # We need v[half_Ns] where N is even.
    # v is defined up to M. half_Ns is up to M//2.
    
    diag = np.zeros(len(Ns), dtype=np.float64)
    diag[even_mask] = v[half_Ns[even_mask]]
    
    scores = (conv_vals + diag) / 2.0
    
    # Find max score
    max_idx = np.argmax(scores)
    best_N = Ns[max_idx]
    best_score = scores[max_idx]
    
    # If best_score is 0, something went wrong.
    if best_score < 1:
        # Fallback to brute force on small range
        # This shouldn't happen with M=10^7
        M_fb = 20000
        powerful_fb = {a * a * b ** 3 for b in range(1, 28) for a in range(1, isqrt(M_fb // b ** 3) + 1)}
        members = set(powerful_fb)
        best_N = 2
        max_pairs = 0
        for n in range(2, M_fb + 1):
            cnt = 0
            for a in powerful_fb:
                if 2 * a > n: break
                if n - a in members:
                    cnt += 1
            if cnt > max_pairs:
                max_pairs = cnt
                best_N = n
        pairs = [[a, best_N - a] for a in sorted(powerful_fb) if 2 * a <= best_N and best_N - a in members]
        return {'N': best_N, 'pairs': pairs}

    # Extract pairs for best_N
    # We need list of a such that a <= best_N - a and a in P and best_N - a in P.
    # a <= best_N // 2
    # a in powerful_list
    # best_N - a in powerful_list
    
    # Use binary search or set lookup
    powerful_set = set(powerful_list)
    
    pairs = []
    # Iterate through powerful numbers <= best_N // 2
    # powerful_list is sorted
    # Find index of last element <= best_N // 2
    # bisect
    import bisect
    limit_a = best_N // 2
    idx = bisect.bisect_right(powerful_list, limit_a)
    
    for a in powerful_list[:idx]:
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': int(best_N), 'pairs': pairs}

# EVOLVE-BLOCK-END
