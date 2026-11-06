import time
import math
import numpy as np

def generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number n = prod p_i^{e_i} with e_i >= 2 for all i.
    Any such number can be written as x^2 * y^3 where y is square-free.
    We iterate over square-free y up to cbrt(limit), and for each y,
    iterate x such that x^2 * y^3 <= limit.
    """
    powerful = set()
    max_y = int(math.ceil(limit ** (1.0/3.0))) + 2

    # Sieve for square-free numbers up to max_y
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
            powerful.add(n)

    return sorted(powerful)

def is_powerful(n):
    """Check if n is a powerful number by exact factorization."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp < 2:
                return False
        p += 1 if p == 2 else 2
    if temp > 1:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Generate powerful numbers up to a limit that allows FFT within time
    # Target L such that 2*L is manageable for rfft
    # |P| ~ C * L^(2/3). For L=5e7, |P| ~ 5e7^(2/3) ~ 1.35e5 (rough)
    # Actually C is about 1.2, so |P| ~ 1.2 * (5e7)^(2/3) ~ 1.2 * 13572 ~ 16k? No.
    # Let's just try L=5e7 and see.
    # Memory for rfft of size 1e8: complex array of 5e7. 5e7 * 16 bytes = 800 MB.
    # This might be tight but numpy is efficient.
    # Let's start with L=2e7 to be safe, then try larger if time permits.
    
    best_pairs = []
    best_n = 0
    
    # Try increasing limits
    limits_to_try = [10**7, 2*10**7, 5*10**7]
    
    for L in limits_to_try:
        if time.time() > deadline - 20:
            break
            
        P = generate_powerful_numbers(L)
        
        if time.time() > deadline - 10:
            break
            
        # Create indicator array
        # We need to compute the number of pairs (a,b) in P with a+b=k, a<=b.
        # Convolution C[k] = sum_{a+b=k} A[a]*A[b].
        # C[k] counts ordered pairs (a,b).
        # For a < b, (a,b) and (b,a) are both counted. So 2 per unordered pair.
        # For a == b, (a,a) is counted once.
        # Let d_k = 1 if k is even and k/2 in P else 0.
        # Number of unordered pairs a<=b is (C[k] + d_k) / 2.
        
        # A is of length 2*L+1 (indices 0 to 2L)
        # But we only care up to 2L.
        # A[i] = 1 if i in P else 0.
        # We can use rfft to save memory/time.
        
        # Size of A: 2*L
        # rfft size: L+1 (for real input of size 2L, output is L+1 complex)
        # Actually, np.fft.rfft(a) where a has size n, returns n//2+1 complex values.
        # We want convolution of A with A.
        # Size of A: 2*L. We want to avoid wrap-around.
        # Since max index is L, and we sum up to 2L, the convolution result size is 2L+1.
        # We can zero-pad A to size 4L to be safe? Or just 2L is enough since support is [1,L].
        # Max sum is 2L. So result index goes up to 2L.
        # A has non-zero at indices 1..L.
        # Convolution C[k] for k up to 2L.
        # If we use FFT size N_fft = 2*L, circular convolution will wrap around.
        # C[k] = sum_{i+j=k} A[i]A[j] + sum_{i+j=k+N_fft} A[i]A[j].
        # Since max index is L, i+j <= 2L. If N_fft=2L, i+j can be 2L.
        # k=0 term gets i+j=2L contribution.
        # So circular convolution is not directly correct unless we pad.
        # Let's pad A to size 4L? That doubles memory.
        # Or, note that for k in [2, 2L-1], there is no wrap-around from i+j >= 2L?
        # i,j <= L. i+j <= 2L.
        # If N_fft = 2L, then i+j = 2L wraps to 0.
        # So C[0] will have the pair (L,L) if 2L is the sum? No, (L,L) sums to 2L.
        # So C[0] gets A[L]*A[L].
        # And C[2L] gets... nothing? No, circular conv: C[k] = sum_{i} A[i] A[(k-i) mod N].
        # This is tricky.
        
        # Safest: Use FFT size >= 2L+1.
        # Let N_fft = 2 * L + 1? Or next power of 2?
        # np.fft.rfft doesn't require power of 2, but it's faster.
        # Let's use next power of 2 >= 2L+1.
        
        target_size = 2 * L + 1
        # Next power of 2
        n_fft = 1
        while n_fft < target_size:
            n_fft *= 2
            
        # If n_fft is too large, break
        if n_fft > 10**8: # 800 MB complex? rfft is half size, so 400MB. OK.
            # Actually rfft output size is n_fft//2 + 1.
            # Memory for n_fft=2^27 ~ 1.3e8. Output ~ 6.7e7 complex. ~1GB.
            # Might be too much.
            # Let's stick to n_fft <= 2^26 ~ 6.7e7. Output ~ 3.3e7. ~500MB.
            if n_fft > 6*10**7:
                continue
                
        # Create A array
        A = np.zeros(n_fft, dtype=np.float64)
        for p in P:
            if p < n_fft:
                A[p] = 1.0
            
        # FFT
        F = np.fft.rfft(A)
        F2 = F * F
        result = np.fft.irfft(F2, n=n_fft)
        
        # result[k] is the circular convolution coefficient for k.
        # Since n_fft >= 2L+1, and support of A is [1,L], the linear convolution is correct for k in [0, 2L].
        # Specifically, for k <= 2L, result[k] is the true linear convolution coefficient.
        
        # Now compute d_k and n_k
        # d_k = 1 if k even and k/2 in P else 0.
        # P_set for fast lookup? Or boolean array?
        # P_bool of size L+1.
        P_bool = np.zeros(L + 1, dtype=np.int8)
        for p in P:
            P_bool[p] = 1
        
        # We are interested in k in [2, 2L].
        # Create d_array of size n_fft? No, just up to 2L.
        # d = np.zeros(2*L+1)
        # For even k, d[k] = P_bool[k//2]
        
        # Vectorized approach
        # indices 2 to 2L
        # even_indices = np.arange(2, 2*L+1, 2)
        # d_vals = P_bool[even_indices // 2]
        
        # We want to find max of (result[k] + d_k) / 2.
        # Let's compute this array.
        
        # To save memory, we can compute on the fly or use a smaller array.
        # result is size n_fft. We only care up to 2L.
        # Let's slice result.
        res_slice = result[:2*L+1]
        
        # Compute d
        # d = np.zeros(2*L+1)
        # even_idx = np.arange(2, 2*L+1, 2)
        # d[even_idx] = P_bool[even_idx // 2]
        
        # We want to maximize (res_slice + d) / 2.
        # This is equivalent to maximizing res_slice + d.
        
        # Let's create d
        d = np.zeros(2*L+1, dtype=np.int32)
        even_idx = np.arange(2, 2*L+1, 2)
        d[even_idx] = P_bool[even_idx // 2]
        
        score = res_slice + d
        
        max_idx = np.argmax(score)
        max_score = score[max_idx]
        
        # Convert to int
        num_pairs = int(max_score) // 2
        
        # Verify num_pairs is reasonable
        if num_pairs > len(best_pairs) and time.time() < deadline - 5:
            # Extract pairs for this N
            N = int(max_idx)
            pairs = []
            P_set = set(P)
            for a in P:
                if a > N // 2:
                    break
                b = N - a
                if b in P_set:
                    pairs.append([a, b])
            
            # Verify all pairs are valid (powerful)
            # The generator guarantees they are powerful.
            # But let's double check a few?
            # For speed, we trust the generator.
            
            if len(pairs) == num_pairs:
                best_pairs = pairs
                best_n = N
            else:
                # Discrepancy? Maybe due to float error in FFT?
                # res_slice should be integer.
                # If mismatch, we can still use the pairs we found, but maybe score is lower?
                # Or maybe FFT error caused wrong max_idx.
                # Let's just update if len(pairs) > len(best_pairs)
                if len(pairs) > len(best_pairs):
                    best_pairs = pairs
                    best_n = N
    
    # If no pairs found, fallback to small N
    if not best_pairs:
        # Small powerful numbers: 1, 4, 8, 9, 16, 25, ...
        # 1+9=10, 4+4=8, 1+4=5
        # Let's find a valid one.
        # 1 and 4 are powerful. 1+4=5. Pairs: [[1,4]]. Score 1.
        # 4 and 4. 4+4=8. Pairs: [[4,4]]. Score 1.
        # 1 and 9. 1+9=10. Pairs: [[1,9]]. Score 1.
        # Let's just return [[1,4]] for N=5.
        best_n = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_n, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
