import time
import numpy as np
from math import isqrt
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # Generate powerful numbers up to a reasonable limit
    # Powerful numbers: every prime factor appears with exponent >= 2
    # Count is ~ C * sqrt(x) for x
    L = 5 * 10**7
    
    powerful_set = set()
    powerful_set.add(1)
    
    # Generate powerful numbers: n = a^2 * b^3 where gcd(a,b) is a square
    # Actually, simpler: n = product of p^e where e >= 2
    # Efficient generation: for each "core" c (squarefree part), n = c^2 * m^2 ... 
    # Simpler: iterate over possible b (cube part) and a (square part)
    # n = a^2 * b^3, but this overcounts. Use the standard generation.
    
    # Standard generation: powerful numbers are of form ∏ p_i^{e_i} with e_i >= 2
    # We can generate by: for each squarefree d, and for each square s, 
    # n = d^2 * s where s is a perfect square and gcd(d, s) = 1... 
    # Actually the simplest correct method:
    # A number n is powerful iff for every prime p|n, p^2|n.
    # Generation: n = a^2 * b^3 where b is squarefree? No.
    # Correct: n = ∏ p_i^{2e_i} for e_i >= 1, so n = (∏ p_i^{e_i})^2 * ∏ p_i^{(e_i mod 2)*...}
    # Let me use: n = a^2 * b^3 where a and b are arbitrary, but this overcounts.
    # The unique representation is: write n = ∏ p_i^{e_i}, e_i >= 2.
    # Let e_i = 2*f_i + g_i where g_i in {0,1}. Then n = (∏ p_i^{f_i + g_i/2})^2... 
    # Simpler approach that's correct:
    # For each possible "cube-free" part c (product of distinct primes), and square s,
    # n = c^2 * s^2 = (c*s)^2... no that's just squares.
    
    # Let me just use the direct method:
    # For b from 1 to L^(1/3), for a from 1 to sqrt(L/b^3):
    #   n = a^2 * b^3
    # But this generates each powerful number multiple times. Use a set.
    
    max_b = int(round(L ** (1/3))) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    powerful = sorted(powerful_set)
    n_powerful = len(powerful)
    
    if time.time() > deadline - 90.0:
        # Fallback
        L = 10**6
        powerful_set = {1}
        max_b = int(round(L ** (1/3))) + 1
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful_set.add(n)
        powerful = sorted(powerful_set)
        n_powerful = len(powerful)
    
    # Now, instead of FFT, let's try candidate N values.
    # Heuristic: N with many divisors might have more representations.
    # But more directly: if N is even, and N/2 is powerful, that's one pair.
    # Let's try: for a range of N values, count representations.
    
    # Strategy: Use FFT but with a smaller, smarter approach.
    # Or: pick N candidates and count directly.
    
    # Let's use FFT for a moderate range and also try some large N candidates.
    
    # First, FFT approach for N up to 2*L
    fft_size = 1
    while fft_size < 2 * L + 1:
        fft_size *= 2
    
    f = np.zeros(fft_size, dtype=np.float64)
    for p in powerful:
        if p < fft_size:
            f[p] = 1.0
    
    f_fft = np.fft.fft(f)
    conv_fft = f_fft * f_fft
    conv = np.fft.ifft(conv_fft).real
    
    # Find best N in [2, 2*L]
    max_N_search = min(2 * L, fft_size - 1)
    Ns = np.arange(2, max_N_search + 1)
    ordered = conv[2:max_N_search + 1]
    mids = Ns // 2
    is_mid_powerful = f[mids] > 0.5
    evens = (Ns % 2 == 0)
    correction = np.where(evens, is_mid_powerful.astype(np.int64), 0)
    ordered_int = np.round(ordered).astype(np.int64)
    unordered = (ordered_int + correction) // 2
    
    max_idx = np.argmax(unordered)
    best_N = int(Ns[max_idx])
    best_count = int(unordered[max_idx])
    
    if time.time() > deadline - 60.0:
        pairs = []
        for a in powerful:
            b = best_N - a
            if b >= a and b in powerful_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    
    # Now try some larger N candidates using direct counting.
    # Candidate N: multiples of small numbers that are "powerful-friendly"
    # Try N = k * M where M is lcm of small squares
    
    # Try N values that are even and large
    # For each candidate N, count pairs by iterating powerful a and checking N-a
    
    # Generate candidate N values
    candidates = set()
    
    # N around 10^15 to 10^18
    # Try multiples of various bases
    base_multipliers = [
        2**10, 2**12, 2**14, 2**16,
        3**6, 3**8, 3**10,
        6**5, 6**6, 6**7,
        12**4, 12**5, 12**6,
        30**4, 30**5,
        60**3, 60**4,
    ]
    
    for bm in base_multipliers:
        # Try N = bm * k for various k
        for k in range(1, 1000):
            N = bm * k
            if N > 10**18:
                break
            if N >= 2:
                candidates.add(N)
    
    # Also try some specific highly composite-like numbers
    for N in [10**15, 2*10**15, 5*10**15, 10**16, 2*10**16, 5*10**16, 10**17, 5*10**17]:
        candidates.add(N)
    
    # Sort candidates and try them
    candidates = sorted(candidates)
    
    best_overall_N = best_N
    best_overall_count = best_count
    best_overall_pairs = None
    
    # Collect best_N pairs first
    pairs_best_N = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs_best_N.append([a, b])
    best_overall_pairs = pairs_best_N
    
    for N in candidates:
        if time.time() > deadline - 30.0:
            break
        
        if N < 2 or N > 10**18:
            continue
        
        # Count pairs: a <= b, a + b = N, a,b powerful
        # a ranges from 1 to N//2
        # But we only have powerful numbers up to L. 
        # For large N, a and b can be up to N, which may exceed L.
        # We need powerful numbers up to N, but we only generated up to L.
        # So this direct counting only works if both a and b <= L, i.e., N <= 2*L.
        # For N > 2*L, we can't use our precomputed set.
        
        # So skip candidates > 2*L for now, or generate more powerful numbers.
        if N > 2 * L:
            continue
        
        count = 0
        pairs = []
        for a in powerful:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall_N = N
            best_overall_pairs = pairs
    
    if best_overall_pairs is None:
        best_overall_pairs = []
    
    return {'N': best_overall_N, 'pairs': best_overall_pairs}
