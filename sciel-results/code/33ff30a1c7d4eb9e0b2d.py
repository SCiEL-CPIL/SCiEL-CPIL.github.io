import struct

# 4 rows x 128 columns of uint16 counters = 1024 bytes
ROWS = 4
COLS = 128
PRIMES = (
    1035493289,
    775374203,
    1495862053,
    2015145641
)
OFFSETS = (
    31, 199, 555, 1029
)

def _get_index(item, row):
    p = PRIMES[row]
    off = OFFSETS[row]
    return (item * p + off) % COLS

def update(state, item):
    buf = bytearray(state)
    for r in range(ROWS):
        idx = _get_index(item, r)
        byte_offset = (r * COLS + idx) * 2
        val = struct.unpack_from('<H', buf, byte_offset)[0]
        val += 1
        struct.pack_into('<H', buf, byte_offset, val)
    return bytes(buf)

def query(state, item):
    min_val = float('inf')
    for r in range(ROWS):
        idx = _get_index(item, r)
        byte_offset = (r * COLS + idx) * 2
        val = struct.unpack_from('<H', state, byte_offset)[0]
        if val < min_val:
            min_val = val
    if min_val == float('inf'):
        return 0
    return min_val
