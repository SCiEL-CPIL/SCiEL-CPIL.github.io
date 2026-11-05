"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the longest run of consecutive integers with distinct totients below 10^12.
    Strategy:
      1. Precompute primes up to 10^6 (since sqrt(10^12) = 10^6).
      2. Define a segmented sieve function to compute phi(n) for n in [L, R] in O(W log log R).
      3. Search over a set of strategic windows:
         - Small range [1, 100000] to catch any short records.
         - Windows near powers of 10 and near 10^12, where longer runs are statistically likely.
         - Multiple windows near 10^12 to cover the upper boundary.
      4. Within each window, slide to find the maximum run of consecutive integers with distinct phi values.
      5. Return the best (start, length) found.
    """
    import math

    MAX_N = 10**12
    MAX_PRIME = int(math.isqrt(MAX_N)) + 1  # 10^6 + 1

    # --- Step 1: Generate primes up to MAX_PRIME using a simple sieve ---
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes = sieve_primes(MAX_PRIME)

    # --- Step 2: Segmented phi sieve ---
    def compute_phi_segment(L, R):
        """Compute phi(n) for all n in [L, R] using segmented sieve."""
        W = R - L + 1
        vals = list(range(L, R + 1))
        res = vals.copy()  # res[i] starts as L+i, we'll modify it

        for p in primes:
            if p * p > R:
                break
            # Find the first number in [L, R] divisible by p
            start = ((L + p - 1) // p) * p
            if start < L:
                start += p
            for j in range(start, R + 1, p):
                idx = j - L
                if res[idx] % p == 0:
                    res[idx] -= res[idx] // p
                    while vals[idx] % p == 0:
                        vals[idx] //= p

        # Handle remaining prime factors > sqrt(R)
        for i in range(W):
            if vals[i] > 1:
                res[i] -= res[i] // vals[i]

        return res

    # --- Step 3: Find longest run in a given phi array ---
    def find_best_run(phi_vals):
        """Given a list of phi values for consecutive integers starting at L,
        find the longest subsequence of consecutive entries with all distinct values."""
        best_len = 0
        best_pos = 0  # 0-indexed within the segment
        n = len(phi_vals)
        for start in range(n):
            seen = set()
            length = 0
            for end in range(start, n):
                v = phi_vals[end]
                if v in seen:
                    break
                seen.add(v)
                length += 1
            if length > best_len:
                best_len = length
                best_pos = start
        return best_len, best_pos

    # --- Step 4: Define search windows ---
    windows = []

    # Small range to catch any short records
    windows.append((1, 100000))

    # Windows near powers of 10
    for k in range(3, 13):
        base = 10 ** k
        wsize = 100000
        L = max(1, base - wsize // 2)
        R = min(MAX_N, base + wsize // 2)
        windows.append((L, R))

    # Windows near 10^12 (upper boundary)
    wsize = 200000
    for offset in range(0, 1000000, wsize):
        L = MAX_N - offset - wsize + 1
        R = MAX_N - offset
        if L >= 1:
            windows.append((L, R))

    # Additional windows at strategic points
    # Multiples of primorials (product of first n primes)
    primorials = [2, 6, 30, 210, 2310, 30030, 510510, 9699690]
    for pm in primorials:
        # Look near multiples of pm that are within [1, MAX_N]
        for mult in [1, 2, 3]:
            base = pm * mult
            if base > MAX_N:
                break
            wsize = 50000
            L = max(1, base - wsize // 2)
            R = min(MAX_N, base + wsize // 2)
            windows.append((L, R))

    # --- Step 5: Search all windows ---
    best_k = 0
    best_start = 1

    for L, R in windows:
        if L < 1:
            L = 1
        if R > MAX_N:
            R = MAX_N
        if L > R:
            continue
        if R - L + 1 > 500000:  # Skip very large windows to stay within time
            continue
        phi_vals = compute_phi_segment(L, R)
        k, pos = find_best_run(phi_vals)
        if k > best_k:
            best_k = k
            best_start = L + pos

    # Ensure we have at least a valid answer
    if best_k < 1:
        best_k = 1
        best_start = 1

    return {'start': best_start, 'length': best_k}
# EVOLVE-BLOCK-END
