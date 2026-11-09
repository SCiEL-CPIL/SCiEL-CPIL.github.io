"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

_NUM_ROWS = 8
_NUM_COLS = 64

# Hash multipliers for better distribution
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    3266489917,
    668265263,
    374761393,
    2097151,
]

def _compute_indices(item):
    """Compute the 8 column indices for the given item."""
    indices = []
    for row in range(_NUM_ROWS):
        idx = ((item * _PRIMES[row] + 31 * row) % 2147483647) % _NUM_COLS
        indices.append(idx)
    return indices

def update(state, item):
    """Update the sketch state by incrementing counters for the given item."""
    counters = list(struct.unpack('<' + 'H' * (_NUM_ROWS * _NUM_COLS), state))
    indices = _compute_indices(item)
    for row, col in enumerate(indices):
        pos = row * _NUM_COLS + col
        counters[pos] = min(counters[pos] + 1, 65535)
    return struct.pack('<' + 'H' * (_NUM_ROWS * _NUM_COLS), *counters)

def query(state, item):
    """Query the estimated frequency of the given item."""
    counters = struct.unpack('<' + 'H' * (_NUM_ROWS * _NUM_COLS), state)
    indices = _compute_indices(item)
    min_val = 65535
    for row, col in enumerate(indices):
        pos = row * _NUM_COLS + col
        if counters[pos] < min_val:
            min_val = counters[pos]
    return min_val
# EVOLVE-BLOCK-END
