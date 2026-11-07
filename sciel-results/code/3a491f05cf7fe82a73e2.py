import struct

# State layout:
# Bytes 0-255: uint8 exact counters for items 0-255
# Bytes 256-1023: 12 rows x 64 columns of uint8 CM sketch for items 256-399
# Total: 256 + 768 = 1024 bytes

_EXACT_END = 256  # items 0..255 use exact counting
_SKETCH_OFFSET = 256
_NUM_ROWS = 12
_NUM_COLS = 64

# Hash parameters for CM sketch (for items 256-399)
_PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641,
    1234567891, 987654323, 314159265, 271828183,
    410332281, 524287001, 628318531, 734003371
]
_SEEDS = [
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79
]
_MOD = 2147483647

def update(state, item):
    buf = bytearray(state)
    
    if item < _EXACT_END:
        # Exact counting for items 0-255
        val = buf[item]
        if val < 0xFF:
            val += 1
        buf[item] = val
    else:
        # CM sketch for items 256-399
        for row in range(_NUM_ROWS):
            p = _PRIMES[row]
            s = _SEEDS[row]
            col = ((item * p + s) % _MOD) % _NUM_COLS
            offset = _SKETCH_OFFSET + row * _NUM_COLS + col
            
            val = buf[offset]
            if val < 0xFF:
                val += 1
            buf[offset] = val
    
    return bytes(buf)

def query(state, item):
    if item < _EXACT_END:
        return state[item]
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
