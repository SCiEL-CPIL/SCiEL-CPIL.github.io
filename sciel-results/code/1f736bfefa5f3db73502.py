"""Hybrid exact/sketch frequency estimator using 1024 bytes of state.

Structural design:
  - Items 0..127: exact uint8 counters (128 bytes). This captures the bulk of
    Zipf(1.6) mass where item 0 has the highest frequency.
  - Items 128..399: 4-row x 168-slot Count-Min Sketch with uint8 counters
    (4 * 168 = 672 bytes). Total = 128 + 672 = 800 bytes <= 1024.

  For the sketch portion, we use the original hash structure but with width 168
  instead of 64, reducing collision probability by ~2.6x per row.

  Query for sketch items returns the minimum of the 4 row counters, which
  bounds the over-estimation.

  Additionally, we apply a small bias correction: since the total stream length
  is 1024 and we know the number of items in the sketch region, we can estimate
  the expected collision floor. However, since we don't track per-item true counts
  in the state, we use a simpler correction: subtract a small constant estimated
  from the average collision rate.

  Actually, let's keep it simpler and just use the wider sketch. The key structural
  change is: exact counters for high-frequency items + wider sketch for the tail.
"""

# Constants
MOD = 2147483647
EXACT_BOUND = 128  # items 0..127 use exact counters
SKETCH_START_ITEM = 128  # items 128..399 use sketch
N_ROWS = 4
W = 168  # width for sketch rows in uint8 counters
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
OFFSETS = (31, 63, 97, 129)  # offsets for each row to decorrelate indices

# State layout:
#   Bytes 0..127: exact uint8 counters for items 0..127
#   Bytes 128..800: 4 rows of 168 uint8 counters for items 128..399
#   Total: 800 bytes


def _sketch_idx(item, row):
    """Compute sketch index for an item in a given row."""
    return (item * PRIMES[row] + OFFSETS[row]) % MOD % W


def update(state, item):
    """Increment the appropriate counter(s) for item in the 1024-byte state."""
    ba = bytearray(state)
    
    if item < EXACT_BOUND:
        # Exact counter: just increment byte at position 'item'
        if ba[item] < 255:
            ba[item] += 1
    else:
        # Use sketch for items >= 128
        for r in range(N_ROWS):
            idx = _sketch_idx(item, r)
            offset = EXACT_BOUND + r * W + idx
            if ba[offset] < 255:
                ba[offset] += 1
    
    return bytes(ba)


def query(state, item):
    """Return nonnegative frequency estimate for item."""
    if item < EXACT_BOUND:
        return float(state[item])
    else:
        best = 255
        for r in range(N_ROWS):
            idx = _sketch_idx(item, r)
            offset = EXACT_BOUND + r * W + idx
            val = state[offset]
            if val < best:
                best = val
        return float(best)
