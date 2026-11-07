import struct

_NUM_ITEMS = 400
_BITS_PER_ITEM = 11
_TOTAL_BITS = _NUM_ITEMS * _BITS_PER_ITEM
_TOTAL_BYTES = (_TOTAL_BITS + 7) // 8

# Precompute offsets
_OFFSETS = []
for i in range(_NUM_ITEMS):
    pos = i * _BITS_PER_ITEM
    byte_idx = pos // 8
    bit_idx = pos % 8
    _OFFSETS.append((byte_idx, bit_idx))

def _read_val(buf, byte_idx, bit_idx):
    val = 0
    # Read 11 bits
    # Bit 0 is LSB
    for k in range(_BITS_PER_ITEM):
        b = byte_idx + (bit_idx + k) // 8
        o = (bit_idx + k) % 8
        if (buf[b] >> o) & 1:
            val |= (1 << k)
    return val

def _write_val(buf, byte_idx, bit_idx, val):
    for k in range(_BITS_PER_ITEM):
        b = byte_idx + (bit_idx + k) // 8
        o = (bit_idx + k) % 8
        if (val >> k) & 1:
            buf[b] |= (1 << o)
        else:
            buf[b] &= ~(1 << o)

def update(state, item):
    buf = bytearray(state)
    byte_idx, bit_idx = _OFFSETS[item]
    val = _read_val(buf, byte_idx, bit_idx)
    val += 1
    _write_val(buf, byte_idx, bit_idx, val)
    return bytes(buf)

def query(state, item):
    byte_idx, bit_idx = _OFFSETS[item]
    val = _read_val(state, byte_idx, bit_idx)
    return float(val)
