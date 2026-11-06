import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Strategy: use FFT to count sums of powerful numbers.
    # Generate powerful numbers up to L, build a boolean indicator array,
    # convolve it with itself to count pairs (a,b) with a+b=N.
    # Then pick the N with the maximum count and retrieve the actual pairs.

    # L = 4 * 10^7 -> 2L = 8 * 10^7 -> next power of 2 = 2^27 = 134,217,728
    # This is manageable for FFT in numpy.
    # Number of powerful numbers up to 4e7 is roughly 2.17*sqrt(4e7) ≈ 2.17*6325 ≈ 13700
    # K^2 ≈ 1.9e8, which is too slow for brute force in Python, but FFT handles it.

    L = 4 * 10**7

    # 1. Generate powerful numbers <= L
    powerful_set = set()
    max_y = int(round(L ** (1/3))) + 2  # y^3 <= L means y <= L^(1/3)

    # Sieve for square-free numbers up to max_y
    # A number is square-free if no p^2 divides it for any prime p.
    # Since max_y is small (~342 for 4e7), we can do trial division by primes.
    # First, get primes up to sqrt(max_y) for the trial division.
    def get_primes(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    small_primes = get_primes(int(math.isqrt(max_y)) + 1)

    sq_free = []
    for y in range(1, max_y + 1):
        is_sq_free = True
        temp_y = y
        for p in small_primes:
            if p * p > temp_y:
                break
            if temp_y % (p * p) == 0:
                is_sq_free = False
                break
        if is_sq_free:
            sq_free.append(y)

    for y in sq_free:
        y_cubed = y ** 3
        if y_cubed > L:
            continue
        max_x = int(math.isqrt(L // y_cubed))
        for x in range(1, max_x + 1):
            val = x * x * y_cubed
            powerful_set.add(val)

    P = sorted(powerful_set)
    K = len(P)

    # Time check
    if time.time() > deadline - 30:
        # Fallback to brute force with smaller L if generation was slow
        # But this shouldn't happen for L=4e7
        pass

    # 2. Use FFT to count sums
    # Create boolean array of size L+1 where arr[i] = 1 if i is powerful
    arr = np.zeros(L + 1, dtype=np.float64)
    for val in P:
        arr[val] = 1.0

    # Convolve arr with itself: result[n] = number of pairs (a,b) with a+b=n
    # Need size at least 2L+1 for the convolution
    # Use FFT for efficient convolution
    fft_size = 1
    while fft_size < 2 * L + 1:
        fft_size <<= 1

    # Perform FFT convolution
    F = np.fft.rfft(arr, n=fft_size)
    F2 = F * F
    conv = np.fft.irfft(F2, n=fft_size)

    # conv[n] should be an integer (count of pairs), but due to floating point
    # it may be slightly off. Round to nearest integer.
    conv_int = np.round(conv[:2*L + 1]).astype(np.int32)

    # 3. Find the N with maximum count
    max_count = np.max(conv_int)
    best_N = int(np.argmax(conv_int))

    if time.time() > deadline - 10:
        # Very tight, just return best found so far
        pass

    # 4. Retrieve actual pairs for verification
    powerful_set_lookup = powerful_set
    pairs = []
    for a in P:
        b = best_N - a
        if b < a:
            continue
        if b <= L and b in powerful_set_lookup:
            pairs.append([a, b])
        # Note: b could be > L, but since a <= L and a + b = N <= 2L,
        # b = N - a >= N - L. Since N <= 2L, b <= 2L - 1 (if a=1).
        # But we only generated powerful numbers up to L.
        # Wait, if N is close to 2L, then b could be close to 2L, which is > L.
        # We need to handle this. The powerful numbers above L up to 2L might be needed.
        # Let's check: if a + b = N and a <= L, then b = N - a.
        # Since N <= 2L, b <= 2L. If b > L, we didn't generate it.
        # This is a problem. We need powerful numbers up to 2L, or at least
        # we need to check if b is powerful even if b > L.
        # But generating powerful numbers up to 2L would double the array size.
        # Alternative: for the pairs we need, b could be up to 2L.
        # We can generate powerful numbers up to 2L.
        # Let's reconsider: the convolution counts pairs where both a and b are in [0, L].
        # But the task allows a and b up to 10^18, with a+b = N <= 10^18.
        # Our approach assumes a, b <= L, so N <= 2L.
        # For pairs where b > L, we miss them in the convolution.
        # To fix: generate powerful numbers up to 2L, and use array size 2L+1.
        # But 2L = 8e7, array size 8e7+1, FFT size 2^27 = 1.34e8. Still feasible.
        # Let's regenerate with L2 = 2L for the powerful number generation.

    # Actually, let's redo with L_gen = 2L to be safe for pairs.
    # But that doubles memory. Let's think:
    # If we set L_gen = 2 * 4e7 = 8e7, then array size is 8e7+1.
    # FFT size for convolution to get sums up to 2*8e7 = 1.6e8 -> 2^28 = 2.68e8.
    # That's a lot of memory for FFT (complex array of size 1.34e8, each 16 bytes -> ~2GB).
    # Too much.
    # Better approach: keep L = 4e7 for generation, but for pair retrieval,
    # we need to check if b is powerful even if b > L.
    # We can write a function to check if a number is powerful by factorization.
    # Since we only do this for the best_N's pairs, and K is ~13700, it's fast.

    def is_powerful(n):
        if n <= 1:
            return True
        temp = n
        for p in small_primes:  # Need primes up to sqrt(n), but n can be up to 2L=8e7
            if p * p > temp:
                break
            if temp % (p * p) == 0:
                # Check if the exponent of p in temp is >= 2
                # Actually, we need to check that every prime factor has exponent >= 2
                # So we divide out all powers of p and check the exponent
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                if exp < 2:
                    return False
        if temp > 1:
            # temp is a prime factor with exponent 1
            return False
        return True

    # We need primes up to sqrt(2L) for the is_powerful function
    # 2L = 8e7, sqrt(8e7) ≈ 8944
    primes_for_check = get_primes(10000)

    def is_powerful_check(n):
        if n <= 1:
            return True
        temp = n
        for p in primes_for_check:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                if exp < 2:
                    return False
        if temp > 1:
            return False
        return True

    # Redo pair retrieval with is_powerful_check
    pairs = []
    for a in P:
        b = best_N - a
        if b < a:
            continue
        if b <= 0:
            continue
        if is_powerful_check(b):
            pairs.append([a, b])

    # Sort pairs by a for consistency
    pairs.sort(key=lambda x: x[0])

    return {'N': best_N, 'pairs': pairs}
