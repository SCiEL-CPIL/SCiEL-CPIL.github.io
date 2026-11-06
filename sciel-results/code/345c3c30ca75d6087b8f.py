"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_NUM_PRIMES = len(_SMALL_PRIMES)
_MASK_SIZE = 1 << _NUM_PRIMES  # 512

def construct():
    start_time = time.time()
    deadline = start_time + 85

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute bitmask for each number: bit j set if _SMALL_PRIMES[j] divides it
    masks = np.zeros(nvals, dtype=np.int32)
    arr = np.arange(lower, max_val + 1, dtype=np.int64)

    for j, p in enumerate(_SMALL_PRIMES):
        bit = 1 << j
        masks[arr % p == 0] |= bit

    mask_list = masks.tolist()

    # Precompute which bit positions are set in each possible mask value (0..511)
    mask_bits = []
    for m in range(_MASK_SIZE):
        bits = []
        for j in range(_NUM_PRIMES):
            if m & (1 << j):
                bits.append(j)
        mask_bits.append(bits)

    # Per-bit frequency counts for the sliding window of 24 numbers
    bit_count = [0] * _NUM_PRIMES

    # Initialize window for n=lower: positions 1..24 (i.e., n+1 to n+24)
    for i in range(1, 25):
        m = mask_list[i]
        for j in mask_bits[m]:
            bit_count[j] += 1

    best_n = lower
    best_count = sum(1 for j in range(_NUM_PRIMES) if bit_count[j] > 0)

    if best_count == 1:
        return best_n

    # Slide the window across all valid n
    for offset in range(1, upper - lower + 1):
        if time.time() > deadline:
            break

        # Remove position offset (leaving the window)
        m_out = mask_list[offset]
        for j in mask_bits[m_out]:
            bit_count[j] -= 1

        # Add position offset + 24 (entering the window)
        m_in = mask_list[offset + 24]
        for j in mask_bits[m_in]:
            bit_count[j] += 1

        # Count distinct primes present (bits with count > 0)
        distinct = 0
        for j in range(_NUM_PRIMES):
            if bit_count[j] > 0:
                distinct += 1

        if distinct < best_count:
            best_count = distinct
            best_n = lower + offset
            if best_count == 1:
                return best_n

    return best_n
# EVOLVE-BLOCK-END
