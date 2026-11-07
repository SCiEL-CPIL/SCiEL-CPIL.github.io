"""Hybrid: 400 exact 8-bit counters + 16-row x 19-col 16-bit Count-Min sketch.

Layout (1024 bytes total):
- Bytes 0..399: exact 8-bit counters for items 0..399 (capped at 255).
- Bytes 400..1023: 16 rows x 19 cols of 16-bit CM sketch (16*19*2 = 608 bytes, 16 bytes unused).

For items with count < 255, the exact byte is returned (zero error).
For the rare overflow case (count >= 255), the CM sketch minimum is used.
"""

NUM_EXACT = 400
SKETCH_OFFSET = 400
NUM_ROWS = 16
NUM_COLS = 19

# 16 distinct 32-bit salts for the CM sketch rows
SALT = (
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
    0x165667b1,
    0x9e3779b9,
    0x85ebca77,
    0xc2b2ae3d,
    0x27d4eb31,
    0x45d9f3b1,
    0x27f3b1c2,
    0x9e3779b1,
    0x165667b3,
    0x85ebca69,
    0xc2b2ae37,
    0x27d4eb2d,
    0x165667b5,
)


def _hash_idx(item, row):
    """Bit-mixing hash to map item to column index in [0, NUM_COLS)."""
    h = (item + 1) * SALT[row]
    h = h & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h % NUM_COLS


def update(state, item):
    ba = bytearray(state)

    # Update exact 8-bit counter (capped at 255)
    if ba[item] < 255:
        ba[item] += 1

    # Update CM sketch for overflow fallback
    for r in range(NUM_ROWS):
        idx = _hash_idx(item, r)
        off = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        val = ba[off] | (ba[off + 1] << 8)
        val += 1
        if val > 65535:
            val = 65535
        ba[off] = val & 0xFF
        ba[off + 1] = (val >> 8) & 0xFF

    return bytes(ba)


def query(state, item):
    exact_val = state[item]

    if exact_val < 255:
        return float(exact_val)

    # Overflow: use CM sketch minimum
    min_val = 65535
    for r in range(NUM_ROWS):
        idx = _hash_idx(item, r)
        off = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2
        val = state[off] | (state[off + 1] << 8)
        if val < min_val:
            min_val = val

    return float(min_val)
