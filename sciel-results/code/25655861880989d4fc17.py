import time
import math
from collections import defaultdict

def generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form p^2 * q^3 ... where p, q are primes.
    More generally, if n = prod p_i^{e_i}, then e_i >= 2 for all i.
    We can generate them by iterating over possible structures.
    
    A powerful number can be written as x^2 * y^3 where y is square-free? 
    No, that's not quite right. 
    Let n = a^2 * b^3. If b is square-free, then any prime in b appears with exponent 3 (odd) or higher?
    Actually, a standard way to generate powerful numbers is:
    n = x^2 * y^3 where y is square-free.
    Why? Let n = prod p_i^{e_i}. e_i >= 2.
    If e_i is even, say 2k, it contributes to x^2.
    If e_i is odd, say 2k+1, since e_i >= 2, 2k+1 >= 3. So e_i = 3, 5, 7...
    We can factor out 3 from the odd exponents? 
    If e_i = 3, it is p^3. This is (p)^3.
    If e_i = 5, it is p^5 = p^2 * p^3.
    If e_i = 7, it is p^7 = p^4 * p^3.
    So any powerful number can be written as x^2 * y^3 where y is square-free.
    Proof: Let n = prod p_i^{e_i}. 
    Let y = prod_{e_i odd} p_i. Since e_i >= 2 and odd, e_i >= 3.
    Then n / y^3 = prod_{e_i odd} p_i^{e_i-3} * prod_{e_i even} p_i^{e_i}.
    For e_i odd, e_i-3 is even (and >= 0). For e_i even, e_i is even.
    So n/y^3 is a perfect square. Let x^2 = n/y^3.
    Then n = x^2 * y^3.
    Also y is square-free by construction.
    
    So we iterate over square-free y, and for each y, iterate over x such that x^2 * y^3 <= limit.
    """
    powerful = set()
    # y must be square-free.
    # Max y such that y^3 <= limit => y <= limit^(1/3)
    max_y = int(round(limit ** (1/3))) + 2
    
    # Generate square-free numbers up to max_y
    # We can use a sieve-like approach or just check.
    # Since max_y is around 10^6 for limit 10^18, checking square-freeness is fast.
    
    def is_square_free(n):
        if n <= 1:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    # Optimize: Precompute primes up to sqrt(max_y) to check square-freeness faster?
    # Or just simple check. max_y ~ 10^6. sqrt(10^6) = 1000.
    # 10^6 * 1000 = 10^9 operations? Might be too slow in Python.
    
    # Better: Sieve for square-free numbers.
    # is_square_free[n] = 1 if square free, 0 otherwise.
    is_sq_free = [True] * (max_y + 1)
    is_sq_free[0] = False
    i = 2
    while i * i <= max_y:
        if is_sq_free[i]: # i is prime? No, we need to mark multiples of i^2.
            # Actually, we just need to mark non-square-free.
            pass
        i += 1
    
    # Let's use a different sieve.
    # Mark multiples of p^2 as not square-free.
    sq_free = [True] * (max_y + 1)
    sq_free[0] = False
    p = 2
    while p * p <= max_y:
        if sq_free[p]: # p is prime (not marked by smaller prime square)
            sq = p * p
            for j in range(sq, max_y + 1, sq):
                sq_free[j] = False
        p += 1
        
    # Now generate powerful numbers
    for y in range(1, max_y + 1):
        if not sq_free[y]:
            continue
        y3 = y * y * y
        if y3 > limit:
            break
        # x^2 <= limit / y3
        max_x = int(math.isqrt(limit // y3))
        for x in range(1, max_x + 1):
            n = x * x * y3
            powerful.add(n)
            
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # We want to maximize the number of pairs.
    # The number of powerful numbers up to X is approx C * X^(2/3).
    # For X = 10^18, count is approx 10^12. Too many to store in memory or iterate over in Python.
    # We need to limit the size of the set P.
    # Let's generate powerful numbers up to a smaller limit L, such that the set size is manageable (e.g., 10^5 or 10^6).
    # Then we find the best sum N from pairs in P.
    # The sum N will be up to 2L.
    # If we pick L such that |P| ~ 10^5, then L ~ (10^5)^(3/2) = 10^7.5 ~ 3*10^7.
    # If we pick L such that |P| ~ 10^6, then L ~ (10^6)^(3/2) = 10^9.
    
    # Let's estimate the time to generate P.
    # Generating P up to 10^9:
    # max_y = 1000.
    # For each y (1000 values), max_x = sqrt(10^9 / y^3).
    # If y=1, max_x = 31622.
    # If y=1000, max_x = 1.
    # Total iterations sum_{y=1}^{1000} sqrt(10^9 / y^3) ~ 10^4.5 * sum y^-1.5 ~ 3*10^4 * 2 ~ 6*10^4.
    # This is very fast.
    # So we can generate P up to 10^12?
    # max_y = 10^4.
    # sum_{y=1}^{10^4} sqrt(10^12 / y^3) ~ 10^6 * 2 ~ 2*10^6.
    # Still fast.
    # Size of P for L=10^12: approx 10^(2/3 * 12) = 10^8. Too big for a set in Python? 
    # 10^8 integers in a set takes ~ 4-8 GB of RAM. Might be too much.
    
    # Let's aim for |P| ~ 10^6.
    # L ~ 10^9.
    # Let's try L = 10^9.
    
    L = 10**9
    P = generate_powerful_numbers(L)
    
    # Check time
    if time.time() > deadline:
         # Fallback to smaller set if generation is too slow (unlikely for 10^9)
         L = 10**7
         P = generate_powerful_numbers(L)

    # Now find the N with the most representations a+b=N, a,b in P, a<=b.
    # We can use a dictionary to count sums.
    # However, P has size ~ 10^6. Pairs are ~ 10^12. Too many to iterate.
    
    # Alternative: 
    # The number of representations r(N) = sum_{a in P, a <= N/2} 1_{N-a in P}.
    # We want to maximize r(N).
    # We can iterate over candidate Ns.
    # What are the candidate Ns? Sums of two elements in P.
    # There are O(|P|^2) sums.
    # But we can be smarter.
    # Note that powerful numbers are sparse.
    # Maybe we can just pick a specific N known to have many representations?
    # Or use a heuristic search.
    
    # Let's try a different approach:
    # Iterate over a in P.
    # For each a, the possible b are in P.
    # This is still O(|P|^2).
    
    # Can we use FFT?
    # Let A(x) = 1 if x in P else 0.
    # The number of pairs (a,b) with a+b=N is the coefficient of x^N in A(x)^2.
    # We can compute the convolution of A with itself using FFT.
    # The length of A is L+1 (indices 0..L).
    # L = 10^9. Array of size 10^9 is too big for FFT in Python (memory and time).
    
    # So we must limit L further.
    # If L = 10^6, |P| ~ 10^4. FFT size 2*10^6. Fast.
    # If L = 10^7, |P| ~ 10^5. FFT size 2*10^7. Maybe okay?
    # 2*10^7 complex numbers ~ 320 MB. Might be tight.
    
    # Let's try L = 10^6 first to ensure it works, then try to increase if time permits.
    # Actually, the seed program used limit=20000.
    # We can do much better.
    
    # Let's use the convolution approach with a manageable L.
    # Let's try L = 10**7.
    # Generate P up to 10**7.
    # |P| ~ 10^4.8 ~ 60000.
    # FFT size 2*10**7.
    # numpy.fft.fft is fast.
    
    # But wait, the task says N can be up to 10^18.
    # If we use L=10^7, the max N is 2*10^7.
    # Is it possible that a larger N has more representations?
    # The density of powerful numbers decreases as x^(-1/3).
    # The expected number of representations for N is ~ N^(1/3) / (2 * C^2) ?
    # Actually, r(N) ~ N^(1/3).
    # So larger N should have more representations on average.
    # We want the MAXIMUM.
    # The maximum might be at the largest N we can handle.
    
    # However, we are constrained by the FFT size.
    # If we can't handle L=10^9, we can't check N up to 10^9.
    
    # Alternative strategy for large N:
    # Pick a random large N (e.g., 10^18) and count representations?
    # No, we need to MAXIMIZE.
    
    # Let's look at the structure.
    # Powerful numbers: 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 108, 121, 125, 128, 144, 169, 196, 216, 225, 243, 256, 288, 289, 324, 343, 361, 392, 400, 441, 484, 512, 529, 576, 625, 675, 676, 729, 784, 841, 864, 900, 961, 1000, 1024, 1089, 1152, 1156, 1225, 1296, 1331, 1369, 1444, 1521, 1536, 1568, 1600, 1681, 1728, 1764, 1849, 1936, 1944, 2025, 2116, 2187, 2197, 2209, 2304, 2401, 2500, 2601, 2704, 2744, 2809, 2880, 2916, 3025, 3125, 3136, 3249, 3364, 3456, 3481, 3600, 3721, 3844, 3969, 4096, 4225, 4356, 4489, 4624, 4761, 4900, 5041, 5184, 5329, 5476, 5625, 5776, 5929, 6084, 6241, 6400, 6561, 6724, 6889, 7056, 7225, 7396, 7569, 7744, 7921, 8100, 8281, 8464, 8649, 8836, 9025, 9216, 9409, 9604, 9801, 10000
    
    # Let's try to generate P up to 10^6 and use FFT.
    # Then we get the best N in [2, 2*10^6].
    # Score will be the count for that N.
    
    # Can we do better?
    # What if we use multiple FFTs for different ranges?
    # Or just increase L until FFT fails.
    
    # Let's try L = 10**7.
    # If it times out, fall back to 10**6.
    
    L = 10**7
    P = generate_powerful_numbers(L)
    
    # Create array A of length 2*L + 1
    # A[i] = 1 if i in P else 0
    # Note: P contains 1.
    
    # Check memory: 2*10^7 floats ~ 160 MB.
    # numpy can handle this.
    
    A = [0.0] * (2 * L + 1)
    for p in P:
        if p <= 2 * L:
            A[p] = 1.0
            
    # FFT
    import numpy as np
    A_np = np.array(A)
    F = np.fft.fft(A_np)
    F2 = F * F
    result = np.fft.ifft(F2).real
    
    # The result array contains the number of pairs (a,b) such that a+b=k.
    # Note: This counts (a,b) and (b,a) as distinct if a!=b.
    # The task asks for distinct pairs a <= b.
    # If a < b, the pair (a,b) is counted once in a<=b, but twice in the convolution (a,b) and (b,a).
    # If a == b, the pair (a,a) is counted once in a<=b, and once in the convolution (a,a).
    
    # Let C[k] be the convolution result.
    # C[k] = sum_{a+b=k} 1.
    # This sum is over all a, b in P.
    # Let n_k be the number of pairs (a,b) with a<=b, a+b=k, a,b in P.
    # C[k] = 2 * (number of pairs with a<b) + (number of pairs with a=b).
    # n_k = (number of pairs with a<b) + (number of pairs with a=b).
    # Let d_k be the number of pairs with a=b (i.e., k=2a, a in P).
    # Then C[k] = 2 * (n_k - d_k) + d_k = 2*n_k - d_k.
    # So n_k = (C[k] + d_k) / 2.
    
    # We need to compute d_k for each k.
    # d_k = 1 if k is even and k/2 in P, else 0.
    
    # We want to maximize n_k.
    # So we maximize (C[k] + d_k) / 2.
    
    best_n = 0
    best_k = 0
    best_d = 0
    
    # Iterate over k
    # k ranges from 2 to 2*L.
    # C[k] is result[k].
    
    # Optimization: Only check k where C[k] is large.
    # But we need d_k too.
    
    # Let's just iterate.
    # 2*10^7 iterations in Python might be slow (~1s).
    # We can use numpy to vectorize.
    
    # d_array[k] = 1 if k even and k/2 in P else 0
    d_array = np.zeros_like(result)
    # k is index.
    # We need to check if k/2 is in P.
    # P is a set of ints.
    # We can create a boolean array for P up to L.
    # P_bool[i] = 1 if i in P else 0.
    
    P_bool = np.zeros(L + 1, dtype=int)
    for p in P:
        if p <= L:
            P_bool[p] = 1
            
    # d[k] = P_bool[k//2] if k even else 0
    # Vectorized:
    # even_indices = np.arange(2, 2*L+1, 2)
    # d_array[even_indices] = P_bool[even_indices // 2]
    
    even_indices = np.arange(2, 2 * L + 1, 2)
    d_array[even_indices] = P_bool[even_indices // 2]
    
    # n_k = (result[k] + d_array[k]) / 2
    n_k = (result + d_array) / 2.0
    
    # Find max
    max_idx = np.argmax(n_k)
    best_n = int(n_k[max_idx])
    best_k = int(max_idx)
    
    # Verify and construct pairs
    # We need the actual pairs for best_k.
    # P is a list.
    # We can find a in P such that a <= best_k/2 and best_k - a in P.
    
    P_set = set(P)
    pairs = []
    for a in P:
        if a > best_k // 2:
            break
        b = best_k - a
        if b in P_set:
            pairs.append([a, b])
            
    # Check time
    if time.time() > deadline:
        # If we are running out of time, maybe return a smaller result?
        # But we already have the best_k.
        # The pair construction is O(|P|) which is ~10^5. Fast.
        pass
        
    return {'N': best_k, 'pairs': pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
