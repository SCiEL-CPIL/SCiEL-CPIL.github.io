"""Rewritten program: brute-force search with optimized LPF sieve and symmetry reduction."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 85  # 90s wall-clock limit, 5s margin

    target = 123457
    max_n = target - 1  # 123456

    # Sieve for largest prime factor
    lpf = bytearray(max_n + 1)  # values up to 123456 fit in 4 bytes; bytearray is 0-255, not enough
    # Actually we need values up to 123456, so use array of ints
    import array
    lpf = array.array('I', [0]) * (max_n + 1)
    lpf[1] = 1

    # Sieve: for each prime i, set lpf[j] = i for all multiples j
    # Since we go in increasing order, the last assignment is the largest prime factor
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i

    best_max = float('inf')
    best_a = 1
    best_b = target - 1

    # Search only a in [1, target//2] due to symmetry: max(P(a), P(target-a)) = max(P(target-a), P(a))
    half = target // 2  # 61728
    for a in range(1, half + 1):
        if (a & 0xFF) == 0 and time.time() > deadline:
            break

        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur = pa if pa >= pb else pb

        if cur < best_max:
            best_max = cur
            best_a = a
            best_b = b
            # Theoretical lower bound: since target is odd, one of a,b is even (P>=2),
            # the other is odd (P>=3 if >1, or 1 if =1). So min possible max is 3.
            if best_max <= 3:
                break

    return [best_a, best_b]
# EVOLVE-BLOCK-END
