"""Memory-bounded streaming frequency sketch: hybrid direct + Count-Min."""
import struct
import time

_T0 = time.monotonic()
_DEADLINE = _T0 + 145.0

# Layout:
#   Bytes 0..127:   128 uint8 counters for items 0..127 (exact counts for likely heavy hitters)
#   Bytes 128..1023: 4 rows of 112 uint16 counters (896 bytes) for CMS on remaining items
#
# Total: 128 + 896 = 1024 bytes exactly.

DIRECT_ITEMS = 128
NUM_ROWS = 4
NUM_COUNTERS = 112  # uint16 per row
COUNTER_BYTES = NUM_ROWS * NUM_COUNTERS * 2  # 896
TOTAL_BYTES = DIRECT_ITEMS + COUNTER_BYTES  # 1024

# Hash parameters
_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647


def _idx(item, row):
    """Compute CMS index for an item in a given row."""
    return ((item * _P[row] + 31 * row) % _MOD) % NUM_COUNTERS


def update(state, item):
    """Update the sketch state with a new item occurrence."""
    if time.monotonic() > _DEADLINE:
        return state  # safety: return unchanged

    new_state = bytearray(state)

    if item < DIRECT_ITEMS:
        # Exact count for items 0..127
        new_state[item] = min(255, new_state[item] + 1)
    else:
        # Increment CMS counters for items 128..399
        for row in range(NUM_ROWS):
            idx = _idx(item, row)
            # Counter location: DIRECT_ITEMS + row * NUM_COUNTERS * 2 + idx * 2
            pos = DIRECT_ITEMS + row * NUM_COUNTERS * 2 + idx * 2
            val = struct.unpack_from('<H', new_state, pos)[0]
            val = min(65535, val + 1)
            struct.pack_into('<H', new_state, pos, val)

    return bytes(new_state)


def query(state, item):
    """Return a nonnegative frequency estimate for the given item."""
    if time.monotonic() > _DEADLINE:
        return 0  # safety

    if item < DIRECT_ITEMS:
        # Exact count for items 0..127
        return state[item]
    else:
        # Return the minimum of the 4 row estimates from CMS
        min_est = 65535
        for row in range(NUM_ROWS):
            idx = _idx(item, row)
            pos = DIRECT_ITEMS + row * NUM_COUNTERS * 2 + idx * 2
            val = struct.unpack_from('<H', state, pos)[0]
            if val < min_est:
                min_est = val
        return min_est
