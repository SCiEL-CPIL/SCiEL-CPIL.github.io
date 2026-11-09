"""Hybrid exact/sketch frequency estimator using 1024 bytes of state."""

# Constants
MOD = 2147483647
W = 224  # width for sketch rows
N_ROWS = 4
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
# Offsets for each row to decorrelate indices
OFFSETS = (31, 63, 97, 129)
EXACT_BOUND = 128  # items 0..127 use exact counters


def _sketch_idx(item, row):
    """Compute sketch index for an item in a given row."""
    return (item * PRIMES[row] + OFFSETS[row]) % MOD % W


def update(state, item):
    """Increment the appropriate counter(s) for item in the 1024-byte state.
    
    State layout:
      - Bytes 0..127: exact uint8 counters for items 0..127
      - Bytes 128..1023: 4 rows of 224 uint8 counters for items 128..399
    """
    ba = bytearray(state)
    
    if item < EXACT_BOUND:
        # Exact counter: just increment byte at position 'item'
        if ba[item] < 255:
            ba[item] += 1
    else:
        # Use sketch for items >= 128
        # Adjust item for sketch (remap to 0..271 range conceptually, but hash handles it)
        for r in range(N_ROWS):
            idx = _sketch_idx(item, r)
            offset = 128 + r * W + idx
            if ba[offset] < 255:
                ba[offset] += 1
    
    return bytes(ba)


def query(state, item):
    """Return nonnegative frequency estimate for item.
    
    For items 0..127, return the exact count.
    For items 128..399, return the minimum of the 4 sketch counters.
    """
    if item < EXACT_BOUND:
        return float(state[item])
    else:
        best = 255
        for r in range(N_ROWS):
            idx = _sketch_idx(item, r)
            offset = 128 + r * W + idx
            val = state[offset]
            if val < best:
                best = val
        return float(best)
