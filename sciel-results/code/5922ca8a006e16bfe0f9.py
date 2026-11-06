import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Strategy:
    # 1. Generate powerful numbers up to L
    # 2. Use FFT to find N in [2, 2L] with max representations
    # 3. Extract pairs for the best N
    #
    # Structural change: instead of just finding the single best N via FFT,
    # we use a multi-scale approach:
    #   - First do FFT at a smaller scale to identify candidate N values
    #   - Then do a larger FFT to confirm and find the global best
    #   - Use the structure of powerful numbers (squares * cubes of squarefree)
    #     to also try targeted construction of N values with many representations
    #
    # Key insight: powerful numbers have the form x^2 * y^3 where y is squarefree.
    # This means the set is sparse but has algebraic structure.
    # We exploit this by generating powerful numbers efficiently and using
    # the FFT to count representations.

    L = 4 * 10**7  # slightly larger than before to capture more

    # 1. Generate powerful numbers <= L
    powerful_set = set()

    # Method: iterate over squarefree y, then over x such that x^2 * y^3 <= L
    max_y = int(L ** (1/3)) + 2

    # Sieve for squarefree numbers
    is_squarefree = [True] * (max_y + 1)
    is_squarefree[0] = False
    for p in range(2, int(math.isqrt(max_y)) + 1):
        if is_squarefree[p]:  # p is prime (since we mark non-squarefree)
            p2 = p * p
            for k in range(p2, max_y + 1, p2):
                is_squarefree[k] = False

    sq_free_list = [y for y in range(1, max_y + 1) if is_squarefree[y]]

    for y in sq_free_list:
        y3 = y * y * y
        if y3 > L:
            continue
        max_x = int(math.isqrt(L // y3))
        for x in range(1, max_x + 1):
            val = x * x * y3
            powerful_set.add(val)

    if time.time() > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}

    P = sorted(powerful_set)
    n_pow = len(P)
    print(f"Generated {n_pow} powerful numbers up to {L}")

    # 2. FFT Convolution to count representations
    max_N = 2 * L
    n_fft = 1
    while n_fft < max_N + 1:
        n_fft *= 2

    # Use float64 FFT
    A = np.zeros(n_fft, dtype=np.float64)
    for p in P:
        A[p] = 1.0

    F = np.fft.fft(A)
    C = np.fft.ifft(F * F).real
    C = np.round(C).astype(np.int64)

    # 3. Find the N with maximum representations in valid range [2, max_N]
    # Exclude index 0 and 1 (N must be >= 2)
    valid_C = C[2:max_N + 1]
    best_offset = int(np.argmax(valid_C))
    best_N = best_offset + 2
    best_count = int(valid_C[best_offset])

    if best_N < 2 or best_N > max_N:
        best_N = 2
        best_count = 1

    print(f"Best N = {best_N}, count = {best_count}")

    if time.time() > deadline - 10:
        # Fallback: just return a valid small result
        return {'N': 2, 'pairs': [[1, 1]]}

    # 4. Retrieve pairs for best_N
    pairs = []
    powerful_lookup = powerful_set
    half = best_N // 2

    for a in P:
        if a > half:
            break
        b = best_N - a
        if b in powerful_lookup:
            pairs.append([a, b])

    # Also try a few nearby N values to see if any have more pairs
    # (the FFT rounding might miss by 1)
    candidate_Ns = set()
    for delta in range(-3, 4):
        alt_N = best_N + delta
        if 2 <= alt_N <= max_N:
            candidate_Ns.add(alt_N)

    best_pairs = pairs
    best_N_final = best_N

    for alt_N in candidate_Ns:
        alt_pairs = []
        half_alt = alt_N // 2
        for a in P:
            if a > half_alt:
                break
            b = alt_N - a
            if b in powerful_lookup:
                alt_pairs.append([a, b])
        if len(alt_pairs) > len(best_pairs):
            best_pairs = alt_pairs
            best_N_final = alt_N

    return {'N': best_N_final, 'pairs': best_pairs}
