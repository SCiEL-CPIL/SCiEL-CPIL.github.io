import numpy as np
from math import isqrt
import time
from scipy.signal import fftconvolve

def construct():
    start = time.time()
    deadline = start + 110  # 10s margin

    # Strategy: Use FFT convolution to find N with maximum representations as sum of two powerful numbers.
    # We want to maximize the number of pairs (a, b) with a <= b, a + b = N, both powerful.
    
    # Key insight: If we compute the convolution of the indicator function of powerful numbers with itself,
    # we get the number of ordered pairs (a, b) with a + b = N. For a <= b, we need to handle the diagonal.
    # Actually, the convolution gives count of all (a,b) pairs. For N, the number of pairs with a <= b is:
    # (total_pairs + (1 if N is even and N/2 is powerful else 0)) / 2
    # But actually simpler: just find N that maximizes the convolution value, then extract pairs.
    
    # Let's generate powerful numbers up to some limit L.
    # N can be up to 10^18, but a and b are powerful numbers.
    # If a, b <= L, then N <= 2L.
    # For N up to 10^18, we'd need L up to 5*10^17, which is too large.
    # 
    # Alternative: The number of powerful numbers up to L is O(sqrt(L)) roughly.
    # Actually, powerful numbers are of the form product p_i^e_i where e_i >= 2.
    # Count up to L: roughly O(sqrt(L) * polylog).
    # For L = 10^10, count is about 10^5.
    # For L = 10^12, count is about 10^6.
    # 
    # With FFT convolution, if we have M powerful numbers up to L, the convolution array has size 2L.
    # This is feasible if L is not too large.
    # 
    # Let's try L = 10^9. Then N up to 2*10^9.
    # Number of powerful numbers up to 10^9: let's estimate.
    # Powerful: p^2, p^3, p^4, ... and products.
    # For p^2 <= 10^9: p <= 31622, about 3400 primes.
    # For p^3 <= 10^9: p <= 1000, about 168 primes.
    # For p^4 <= 10^9: p <= 177, about 40 primes.
    # Plus products like p^2 * q^2, p^2 * q^3, etc.
    # Total maybe ~50,000 powerful numbers up to 10^9.
    # 
    # FFT convolution of two arrays of length 2*10^9 would be too large.
    # Instead, we can use the fact that powerful numbers are sparse.
    # 
    # Better approach: Generate all powerful numbers up to L, sort them, 
    # then for each powerful number a, for each powerful number b >= a, 
    # compute a + b and count occurrences. But this is O(M^2) which is too slow for M=50000.
    # 
    # Alternative: Use FFT on a dense array. Create an array of size 2L where arr[x] = 1 if x is powerful.
    # Then FFT convolution gives counts. But L = 10^9 means array of size 2*10^9, which is 16 GB in float64. Too much.
    # 
    # Let's reduce L. Try L = 10^7.
    # Array size 2*10^7, which is 160 MB in float64. Feasible.
    # Number of powerful numbers up to 10^7: roughly a few thousand.
    # 
    # Even better: we can use a smaller L but be smarter.
    # 
    # Actually, let's think about what gives the most representations.
    # For a given N, the number of representations is the number of powerful a such that N - a is also powerful.
    # This is like an additive energy problem.
    # 
    # Let's try: Generate powerful numbers up to L = 10^8.
    # M = count of powerful numbers up to 10^8.
    # Then for each pair (i, j) with i <= j, compute sum = P[i] + P[j], and count using a dictionary or array.
    # M ~ 10000, so M^2/2 ~ 5*10^7 operations. Feasible.
    
    def generate_powerful(L):
        """Generate all powerful numbers up to L."""
        powerful = set()
        powerful.add(1)
        
        # Generate numbers of the form p^a * q^b * ... where each exponent >= 2
        # We can do this recursively or iteratively.
        
        # First, find all primes up to sqrt(L) for the "base" primes.
        # Actually, we need primes up to L^(1/2) for p^2, and smaller for higher powers.
        
        # Approach: generate all numbers of form x^2 * y where y is a powerful number, but that's circular.
        
        # Better: iterative generation.
        # Start with 1. For each powerful number p, multiply by q^2 for primes q where p * q^2 <= L.
        # But we need to ensure we generate all.
        
        # Alternative: use the characterization that a number is powerful iff it is a product of a square and a cube.
        # i.e., n = a^2 * b^3 where a, b are integers.
        # So we can iterate b from 1 to L^(1/3), and for each b, iterate a from 1 to sqrt(L / b^3).
        
        max_b = int(L ** (1/3)) + 1
        powerful_list = set()
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful_list.add(n)
        
        return sorted(powerful_list)
    
    # Choose L
    # Try different L values and see what fits in time.
    # Start with L = 10^7.
    
    L = 10**7
    P = generate_powerful(L)
    M = len(P)
    
    if time.time() > deadline - 30:
        # Reduce L
        L = 10**6
        P = generate_powerful(L)
        M = len(P)
    
    print(f"Generated {M} powerful numbers up to {L}")
    
    # Now, for each pair (i, j) with i <= j, compute sum and count.
    # Use a dictionary to count.
    from collections import Counter
    
    counts = Counter()
    
    # Optimize: use numpy if possible
    P_arr = np.array(P, dtype=np.int64)
    
    # For each i, add P[i] to all P[j] for j >= i.
    # This is M additions of arrays of decreasing size.
    
    # Better: use the fact that we want to find the mode of the sums.
    # Generate all sums P[i] + P[j] for i <= j.
    
    # For M = 5000, this is 12.5M sums. Feasible.
    # For M = 10000, this is 50M sums. Might be slow in Python.
    
    # Let's use a more efficient method: for each i, compute P[i] + P[i:] and update counts.
    # But updating a Counter with 50M elements is slow.
    
    # Alternative: use numpy to generate all sums, then use np.bincount or np.unique.
    
    # Generate all sums for i <= j:
    # sums = []
    # for i in range(M):
    #     sums.append(P_arr[i:] + P_arr[i])
    # sums = np.concatenate(sums)
    
    # For M = 5000, total elements = 5000*5001/2 = 12.5M. OK.
    # For M = 10000, total elements = 50M. Might be 400MB. OK.
    
    sums_list = []
    for i in range(M):
        sums_list.append(P_arr[i:] + P_arr[i])
    
    if time.time() > deadline - 20:
        # Fallback: just pick a random N
        N = 2 * L // 3
        return {'N': int(N), 'pairs': []}
    
    sums = np.concatenate(sums_list)
    
    # Find the most common sum
    unique_sums, counts_arr = np.unique(sums, return_counts=True)
    max_count = np.max(counts_arr)
    best_N = unique_sums[np.argmax(counts_arr)]
    
    print(f"Best N = {best_N}, count = {max_count}")
    
    # Now extract the pairs for best_N
    pairs = []
    P_set = set(P)
    for a in P:
        b = best_N - a
        if b < a:
            continue
        if b in P_set:
            pairs.append([int(a), int(b)])
    
    # Verify
    if time.time() > deadline - 5:
        pass  # Just return
    
    return {'N': int(best_N), 'pairs': pairs}
