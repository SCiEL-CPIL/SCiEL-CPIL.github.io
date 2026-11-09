"""Memory-bounded streaming frequency sketches: 16-bit CM sketch with precomputation."""

# State layout:
# 4 rows of 128 uint16 counters.
# Total size: 4 * 128 * 2 = 1024 bytes.

_ROWS = 4
_SLOTS = 128
_TOTAL_SIZE = _ROWS * _SLOTS * 2

_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
_M = 2147483647

# Precompute indices for all possible items (0..399)
# indices[item][row] -> byte offset of the uint16 counter
_PRECOMPUTED_OFFSETS = []
for item in range(400):
    offsets = []
    for row in range(_ROWS):
        p = _PRIMES[row]
        col = ((item * p + 31 * row) % _M) % _SLOTS
        byte_offset = row * (_SLOTS * 2) + col * 2
        offsets.append(byte_offset)
    _PRECOMPUTED_OFFSETS.append(offsets)

def update(state, item):
    """Update the sketch state with a new item."""
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
    """Query the estimated frequency of an item."""
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
