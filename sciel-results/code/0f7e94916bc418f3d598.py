# Hybrid frequency sketch: exact counters for low-frequency items + CMS for high-frequency
# Layout:
#   Bytes 0-127: exact uint8 counters for items 0-127 (128 bytes)
#   Bytes 128-1023: 4 rows x 224 cols of uint8 CMS for items 128-399 (896 bytes)
#
# Rationale:
#   - Items 0-127 get exact counts (capped at 255). In the uniform stream, each appears
#     ~2.56 times on average, so exact is perfect. In Zipf, items 0-127 have moderate
#     frequency; exact handles them well. In stream 3, items 0-19 get 512/20=25.6 each
#     (fits in uint8), items 150-399 get 512/250=2.05 each.
#   - Items 128-399 use a 4-row x 224-col CMS with uint8 counters. Width 224 gives
#     expected load of 272/224 ≈ 1.21 items per counter, reducing collision bias.
#   - The min-of-4 query gives a tight upper bound on the true count.
#
# Key structural change from seed: the seed used 4x64 uint32 (all items in CMS).
# Here we split: exact for the first 128 items (which covers the high-frequency
# Zipf items and stream-3 items 0-19), and a wider CMS for the remaining 272 items.
# This eliminates all error for the 128 exact items and reduces collision error
# for the CMS items by using width 224 instead of 64.

EXACT_N = 128
W = 224
D = 4
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
PRIME = 2147483647
CMS_START = 128

# Precompute CMS indices for items 128-399
# _CMS_IDX[item - 128] = (idx0, idx1, idx2, idx3) where idx is 0..223
_CMS_IDX = []
for item in range(128, 400):
    row_indices = []
    for r in range(D):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % PRIME) % W
        row_indices.append(idx)
    _CMS_IDX.append(tuple(row_indices))


def update(state, item):
    buf = bytearray(state)

    if item < EXACT_N:
        # Exact counter: byte at position item
        v = buf[item]
        if v < 255:
            buf[item] = v + 1
    else:
        # CMS for items 128-399
        i0, i1, i2, i3 = _CMS_IDX[item - EXACT_N]

        off0 = CMS_START + i0
        v = buf[off0]
        if v < 255:
            buf[off0] = v + 1

        off1 = CMS_START + W + i1
        v = buf[off1]
        if v < 255:
            buf[off1] = v + 1

        off2 = CMS_START + 2 * W + i2
        v = buf[off2]
        if v < 255:
            buf[off2] = v + 1

        off3 = CMS_START + 3 * W + i3
        v = buf[off3]
        if v < 255:
            buf[off3] = v + 1

    return bytes(buf)


def query(state, item):
    if item < EXACT_N:
        # Exact count
        return state[item]
    else:
        # CMS min
        i0, i1, i2, i3 = _CMS_IDX[item - EXACT_N]

        v0 = state[CMS_START + i0]
        v1 = state[CMS_START + W + i1]
        v2 = state[CMS_START + 2 * W + i2]
        v3 = state[CMS_START + 3 * W + i3]

        m = v0
        if v1 < m:
            m = v1
        if v2 < m:
            m = v2
        if v3 < m:
            m = v3

        return m
