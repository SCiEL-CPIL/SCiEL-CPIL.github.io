"""
Hybrid frequency estimator: exact uint8 counters for items 0..199
plus a 3-row x 137-slot uint16 Count-Min sketch for items 200..399.

State layout (1024 bytes):
  bytes 0..199   : uint8 exact counters for items 0..199
  bytes 200..1021: 3 rows of 137 little-endian uint16 counters (822 bytes)
                   for items 200..399 (200 items)
  bytes 1022..1023: unused (zero)
"""

N_EXACT = 200
N_SKETCH = 200
N_ROWS = 3
W = 137

# Primes for hashing
PRIMES = (1035493289, 775374203, 1495862053)

# Pre-compute the 3 indices for each sketch item (200..399)
_SKETCH_IDX = [[0] * N_ROWS for _ in range(N_SKETCH)]
for _i in range(N_SKETCH):
    _item = N_EXACT + _i
    for _r in range(N_ROWS):
        _SKETCH_IDX[_i][_r] = (_item * PRIMES[_r] + 31 * _r) % W


def update(state, item):
    """Increment the counter(s) for item and return the new 1024-byte state."""
    ba = bytearray(state)

    if item < N_EXACT:
        # Exact uint8 counter
        if ba[item] < 255:
            ba[item] += 1
    else:
        # Sketch: 3 rows of uint16 LE counters starting at byte 200
        idx_table = _SKETCH_IDX[item - N_EXACT]
        base = 200
        for r in range(N_ROWS):
            col = idx_table[r]
            off = base + r * W * 2 + col * 2
            # Read uint16 LE
            val = ba[off] | (ba[off + 1] << 8)
            if val < 0xFFFF:
                val += 1
                ba[off] = val & 0xFF
                ba[off + 1] = (val >> 8) & 0xFF

    return bytes(ba)


def query(state, item):
    """Return a non-negative frequency estimate for item."""
    if item < N_EXACT:
        return float(state[item])
    else:
        idx_table = _SKETCH_IDX[item - N_EXACT]
        base = 200
        best = 0xFFFF
        for r in range(N_ROWS):
            col = idx_table[r]
            off = base + r * W * 2 + col * 2
            val = state[off] | (state[off + 1] << 8)
            if val < best:
                best = val
        return float(best)
