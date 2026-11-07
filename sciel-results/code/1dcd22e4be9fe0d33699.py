import struct

# State layout:
# Bytes 0-539: 270 uint16 counters for items 0-19 and 150-399
# Bytes 540-1023: 4 rows x 121 cols uint8 CM sketch for items 20-149
# Total: 540 + 484 = 1024 bytes

_EXACT_ITEMS_LOW = list(range(0, 20))
_EXACT_ITEMS_HIGH = list(range(150, 400))
_SKETCH_ITEMS = list(range(20, 150))

_EXACT_OFFSET = 0
_SKETCH_OFFSET = 540
_NUM_ROWS = 4
_NUM_COLS = 121

# Mapping for exact items to index in the 270-element array
# 0-19 -> 0-19
# 150-399 -> 20-269
def _exact_index(item):
    if item < 20:
        return item
    elif item >= 150:
        return 20 + (item - 150)
    else:
        return -1

_PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641
]
_SEEDS = [
    31, 37, 41, 43
]
_MOD = 2147483647

def update(state, item):
    buf = bytearray(state)
    idx = _exact_index(item)
    
    if idx != -1:
        # 16-bit counter update
        offset = idx * 2
        val = int.from_bytes(buf[offset:offset+2], 'little')
        if val < 0xFFFF:
            val += 1
        buf[offset:offset+2] = val.to_bytes(2, 'little')
    else:
        # CM sketch update (8-bit counters)
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
    idx = _exact_index(item)
    if idx != -1:
        offset = idx * 2
        return int.from_bytes(state[offset:offset+2], 'little')
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
