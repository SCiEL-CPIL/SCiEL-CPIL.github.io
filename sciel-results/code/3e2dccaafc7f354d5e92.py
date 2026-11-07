"""Memory-bounded streaming frequency sketch: Wide Count-Min Sketch."""
import struct
import time

_T0 = time.monotonic()
_DEADLINE = _T0 + 145.0

# Layout: 4 rows of 128 uint16 counters
# Total bytes: 4 * 128 * 2 = 1024
NUM_ROWS = 4
NUM_COUNTERS = 128
COUNTER_SIZE = 2  # uint16
ROW_SIZE = NUM_COUNTERS * COUNTER_SIZE  # 256 bytes per row
TOTAL_BYTES = NUM_ROWS * ROW_SIZE  # 1024 bytes

# Hash parameters (same primes as seed, adjusted modulus)
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

    for row in range(NUM_ROWS):
        idx = _idx(item, row)
        # Counter location: row * ROW_SIZE + idx * COUNTER_SIZE
        pos = row * ROW_SIZE + idx * COUNTER_SIZE
        val = struct.unpack_from('<H', new_state, pos)[0]
        # Max items is 1024, so uint16 (65535) is safe, but cap just in case
        if val < 65535:
            val += 1
        struct.pack_into('<H', new_state, pos, val)

    return bytes(new_state)


def query(state, item):
    """Return a nonnegative frequency estimate for the given item."""
    if time.monotonic() > _DEADLINE:
        return 0  # safety

    min_est = 65535
    for row in range(NUM_ROWS):
        idx = _idx(item, row)
        pos = row * ROW_SIZE + idx * COUNTER_SIZE
        val = struct.unpack_from('<H', state, pos)[0]
        if val < min_est:
            min_est = val
    return min_est
