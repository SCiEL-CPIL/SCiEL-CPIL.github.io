import struct

# State layout:
# Bytes 0-19: exact uint8 counters for items 0-19 (20 items)
# Bytes 20-269: exact uint8 counters for items 150-399 (250 items)
# Bytes 270-1023: 13 rows x 58 cols uint8 CM sketch for items 20-149 (130 items)
# Total: 20 + 250 + 754 = 1024 bytes

_EXACT_LOW_OFFSET = 0
_EXACT_HIGH_OFFSET = 20
_SKETCH_OFFSET = 270
_NUM_ROWS = 13
_NUM_COLS = 58

# Hash parameters for CM sketch (for items 20-149)
_PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641,
    1234567891, 987654323, 314159265, 271828183,
    410332281, 524287001, 628318531, 734003371, 857103881
]
_SEEDS = [
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83
]
_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    
    if item < 20:
        # Exact counter for items 0-19
        offset = _EXACT_LOW_OFFSET + item
        val = buf[offset]
        if val < 0xFF:
            buf[offset] = val + 1
    elif item >= 150:
        # Exact counter for items 150-399
        offset = _EXACT_HIGH_OFFSET + (item - 150)
        val = buf[offset]
        if val < 0xFF:
            buf[offset] = val + 1
    else:
        # CM sketch for items 20-149
        for row in range(_NUM_ROWS):
            p = _PRIMES[row]
            s = _SEEDS[row]
            col = ((item * p + s) % _MOD) % _NUM_COLS
            offset = _SKETCH_OFFSET + row * _NUM_COLS + col
            val = buf[offset]
            if val < 0xFF:
                buf[offset] = val + 1
    
    return bytes(buf)


def query(state, item):
    if item < 20:
        return state[_EXACT_LOW_OFFSET + item]
    elif item >= 150:
        return state[_EXACT_HIGH_OFFSET + (item - 150)]
    else:
        min_val = 0xFF
        for row in range(_NUM_ROWS):
            p = _PRIMES[row]
            s = _SEEDS[row]
            col = ((item * p + s) % _MOD) % _NUM_COLS
            offset = _SKETCH_OFFSET + row * _NUM_COLS + col
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val
