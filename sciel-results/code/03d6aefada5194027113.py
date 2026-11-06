"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from math import isqrt, isperfect

    L = 10_000_000  # search range for N

    # Generate all powerful numbers up to L
    # A powerful number has the form a^2 * b^3 where gcd(a,b)=1
    # Equivalently, every prime factor appears to an even power
    powerful_set = set()
    powerful_set.add(1)  # 1 is powerful

    # Generate numbers of the form a^2 * b^3
    # b^3 <= L, so b <= L^(1/3) ~ 215
    max_b = int(round(L ** (1.0 / 3.0))) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        # a^2 <= L / b^3
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)

    powerful = sorted(powerful_set)
    P = len(powerful)

    # Build binary array and use FFT for convolution
    arr = np.zeros(L + 1, dtype=np.float64)
    for x in powerful:
        arr[x] = 1.0

    # FFT convolution: count[a+b] for all a,b in powerful
    n_fft = 1
    while n_fft < 2 * (L + 1):
        n_fft *= 2
    arr_padded = np.zeros(n_fft, dtype=np.float64)
    arr_padded[:L + 1] = arr
    fft_a = np.fft.rfft(arr_padded)
    conv = np.fft.irfft(fft_a * fft_a, n=n_fft)
    # conv[i] = number of ordered pairs (a,b) with a+b = i
    # We want unordered pairs with a <= b, so for i, unordered = conv[i] // 2 + (1 if i is even and i//2 in powerful_set else 0)
    # Actually, conv[i] counts ordered pairs (a,b). Unordered with a<=b:
    # If a != b: each unordered pair {a,b} contributes 2 to conv
    # If a == b: contributes 1 to conv
    # So unordered_count = (conv[i] + same_count) / 2 where same_count = 1 if i/2 is powerful and i is even

    best_N = -1
    best_count = 0
    for N in range(2, L + 1):
        c = int(round(conv[N]))
        # Count pairs with a <= b, a+b=N, a,b powerful
        # c counts ordered pairs. Pairs with a<b: each contributes 2. Pairs with a=b: contributes 1.
        # Let k = number of pairs with a=b (i.e., N even and N/2 powerful)
        # Then c = 2*(num_distinct) + k
        # Unordered pairs with a<=b = num_distinct + k = (c - k)/2 + k = (c + k)/2
        k = 0
        if N % 2 == 0 and (N // 2) in powerful_set:
            k = 1
        unordered = (c + k) // 2
        if unordered > best_count:
            best_count = unordered
            best_N = N

    # Extract the actual pairs for best_N
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])

    # Verify: ensure all pairs are valid (a <= b, a+b = N, both powerful)
    verified_pairs = []
    for p in pairs:
        a, b = p
        if a <= b and a + b == best_N and a in powerful_set and b in powerful_set:
            verified_pairs.append([a, b])

    # If verification found fewer pairs than expected, still return what we have
    # The score is the number of pairs
    return {'N': best_N, 'pairs': verified_pairs}
# EVOLVE-BLOCK-END
