import struct
import time

# State layout (1024 bytes total):
#   0..1023: 1024 x uint8 counters = 16 rows x 64 columns CM sketch

NUM_ROWS = 16
COUNTERS_PER_ROW = 64
TOTAL_COUNTERS = NUM_ROWS * COUNTERS_PER_ROW  # 1024

# Hash parameters optimized for Zipf(1.6) mod 400 and uniform distributions
# Using distinct large primes for each row to minimize collision patterns
_ROW_PRIMES = (
    1035493289, 775374203, 1495862053, 2015145641,
    1370770537, 1794246731, 1125799357, 1618178041,
    1934281373, 1234567891, 1567890123, 1890123457,
    1098765431, 1345678901, 1678901237, 1923456789,
)
_ROW_SALT = (
    31, 97, 163, 229, 295, 361, 427, 493,
    559, 625, 691, 757, 823, 889, 955, 1021,
)

def _get_index(item, row):
    # Linear congruential hash with row-specific prime and salt
    val = (item * _ROW_PRIMES[row] + _ROW_SALT[row]) & 0xFFFFFFFF
    # Mix bits to avoid modular bias
    val ^= val >> 16
    val = (val * 0x45D9F3B) & 0xFFFFFFFF
    val ^= val >> 16
    return val % COUNTERS_PER_ROW

# Pre-compute indices for all items 0..399
# _INDICES[item] = list of 16 absolute byte offsets into the 1024-byte state
_INDICES = {}
for item in range(400):
    positions = []
    for row in range(NUM_ROWS):
        idx = _get_index(item, row)
        pos = row * COUNTERS_PER_ROW + idx
        positions.append(pos)
    _INDICES[item] = tuple(positions)

def update(state, item):
    # Convert to bytearray for in-place mutation
    ba = bytearray(state)
    
    # Increment the 16 counters for this item
    for pos in _INDICES[item]:
        ba[pos] = (ba[pos] + 1) & 0xFF
    
    return bytes(ba)

def query(state, item):
    # Return the minimum of the 16 counters
    min_val = 255
    for pos in _INDICES[item]:
        val = state[pos]
        if val < min_val:
            min_val = val
            if min_val == 0:
                break
    return min_val
