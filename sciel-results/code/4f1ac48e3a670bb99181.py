"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    n_primes = len(small_primes)
    W = 24
    N_MIN = 1000
    N_MAX = 1000000
    N_END = N_MAX + W  # last m we need a mask for

    # Build mask array: mask[m] = bitmask of which small primes divide m
    mask = [0] * (N_END + 1)
    for j, p in enumerate(small_primes):
        bit = 1 << j
        for m in range(p, N_END + 1, p):
            mask[m] |= bit

    # Sliding window OR with frequency of each mask value
    freq = [0] * 512  # masks are 9-bit, max value 511
    cur_or = 0

    # Initialize window for n = N_MIN - 1 (covers m = N_MIN .. N_MIN+W-1)
    for i in range(W):
        m = N_MIN - 1 + i  # = N_MIN + (i-1), i in 1..W
        # Actually m = n + i where n = N_MIN-1, i in 1..W => m in N_MIN..N_MIN+W-1
        v = mask[m]
        freq[v] += 1
        cur_or |= v

    best_n = N_MIN - 1
    best_count = bin(cur_or).count('1')

    # Slide: for n from N_MIN-1+1 to N_MAX-1
    # Window for n: m = n+1 .. n+W
    # When n increments by 1: remove m = old_n+1, add m = new_n+W
    for n in range(N_MIN, N_MAX):
        if time.time() > deadline:
            break
        # Remove m = n (the leftmost: old window was [n, n+W-1] for old_n=n-1)
        # Wait: for old_n = n-1, window is m = (n-1)+1 .. (n-1)+W = n .. n+W-1
        # For new_n = n, window is m = n+1 .. n+W
        # So remove m = n, add m = n+W
        out_m = n
        in_m = n + W
        v_out = mask[out_m]
        v_in = mask[in_m]
        freq[v_out] -= 1
        freq[v_in] += 1
        # Recompute OR
        if freq[v_out] == 0:
            # This mask value no longer present; need to recompute OR from scratch
            # Only recompute when a value drops to 0
            cur_or = 0
            for mv in range(512):
                if freq[mv]:
                    cur_or |= mv
        else:
            # OR can only gain bits
            cur_or |= v_in

        cnt = bin(cur_or).count('1')
        if cnt < best_count:
            best_count = cnt
            best_n = n
            # If we hit the theoretical lower bound, we can stop
            if best_count <= 4:
                # Can't do better than this in practice; continue briefly to confirm
                pass

    return best_n
# EVOLVE-BLOCK-END
