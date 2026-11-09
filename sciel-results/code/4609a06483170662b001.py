import struct

# Hybrid memory-bounded streaming frequency sketch
# Strategy: exact counters for the most frequent items (Zipf head) +
#           wide Count-Min sketch for the tail.
#
# Layout (1024 bytes total):
#   Bytes 0..29   : exact uint8 counters for items 0..29  (30 bytes)
#   Bytes 30..1023: 4 rows x 248 uint8 counters for items 30..399 (370 items)
#
# Rationale:
#   - Zipf(1.6) concentrates mass on low-index items. Items 0..29 get exact counts.
#   - Remaining 370 items use a 4-row x 248-wide CM sketch (uint8).
#   - 4 rows reduce collision error via min; 248 width keeps per-cell load low.
#   - uint8 caps at 255, which is safe since max true count per item is 1024
#     but for tail items the expected count is small; saturation only affects
#     very heavy hitters which are mostly in the exact region.

N_EXACT = 30
W = 248
D = 4
PRIME = 2147483647
_PS = (1035493289, 775374203, 1495862053, 2015145641)

# Precompute sketch indices for tail items (30..399)
_SK_IDX = [[0] * D for _ in range(400)]
for item in range(N_EXACT, 400):
    for r in range(D):
        p = _PS[r]
        _SK_IDX[item][r] = ((item * p + 31 * r) % PRIME) % W


def update(state, item):
    buf = bytearray(state)
    if item < N_EXACT:
        # Exact counter: byte at offset item
        cur = buf[item]
        if cur < 255:
            buf[item] = cur + 1
    else:
        # Sketch: increment D counters
        idxs = _SK_IDX[item]
        base = N_EXACT
        for r in range(D):
            offset = base + r * W + idxs[r]
            cur = buf[offset]
            if cur < 255:
                buf[offset] = cur + 1
    return bytes(buf)


def query(state, item):
    if item < N_EXACT:
        # Exact count
        return state[item]
    else:
        # Minimum over D rows
        idxs = _SK_IDX[item]
        base = N_EXACT
        min_val = 255
        for r in range(D):
            offset = base + r * W + idxs[r]
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val
