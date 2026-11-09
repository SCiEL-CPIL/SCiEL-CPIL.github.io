"""Memory-bounded streaming frequency sketches: 2-row x 256-col uint16 CM sketch.

Structural change from 4x128 to 2x256: with only 400 distinct items, wider rows
reduce collision probability per row. Expected load per counter drops from 3.125
to 1.56, meaning most counters hold 0-1 items. The min-of-2 on wide counters
provides tighter bounds than min-of-4 on narrow counters for this domain.
"""

_ROWS = 2
_SLOTS = 256
_TOTAL_SIZE = _ROWS * _SLOTS * 2  # 1024 bytes

# Hash parameters: (multiplier, offset) per row
# Chosen to maximize spread across 256 slots for items 0..399
_PARAMS = (
    (1035493289, 31),
    (775374203, 97),
)
_M = 2147483647  # 2^31 - 1

# Precompute byte offsets for all possible items (0..399)
# offsets[item] -> list of byte offsets (one per row)
_PRECOMPUTED_OFFSETS = []
for item in range(400):
    offsets = []
    for row in range(_ROWS):
        a, b = _PARAMS[row]
        col = ((item * a + b) % _M) % _SLOTS
        byte_offset = row * (_SLOTS * 2) + col * 2
        offsets.append(byte_offset)
    _PRECOMPUTED_OFFSETS.append(offsets)


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    if not isinstance(state, bytes) or len(state) != _TOTAL_SIZE:
        return state

    if item < 0 or item > 399:
        return state

    buf = bytearray(state)
    offsets = _PRECOMPUTED_OFFSETS[item]

    for offset in offsets:
        # Read uint16 little-endian
        val = buf[offset] | (buf[offset + 1] << 8)
        if val < 65535:
            val += 1
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF

    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    if not isinstance(state, bytes) or len(state) != _TOTAL_SIZE:
        return 0.0

    if item < 0 or item > 399:
        return 0.0

    offsets = _PRECOMPUTED_OFFSETS[item]
    min_val = 65535

    for offset in offsets:
        val = state[offset] | (state[offset + 1] << 8)
        if val < min_val:
            min_val = val

    return float(min_val)
