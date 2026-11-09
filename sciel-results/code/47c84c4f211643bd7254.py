import struct

# Strategy: Tiered Precision Allocation
# Tier 1: Items 0..15 (16 items) -> 4 bytes (uint32) each. Total 64 bytes.
#         Exact counts for the highest frequency Zipf items.
# Tier 2: Items 16..255 (240 items) -> 1 byte (uint8) each. Total 240 bytes.
#         Exact counts for mid/low frequency items. Overflow protection: cap at 255.
# Tier 3: Items 256..399 (144 items) -> Count-Min Sketch.
#         Remaining bytes: 1024 - 64 - 240 = 720 bytes.
#         720 / 2 = 360 uint16 counters.
#         Configuration: 4 rows x 90 columns.
#         Hashing: map item (256..399) to 0..143, then hash to 0..89.

_T1_SIZE = 16
_T2_START = 16
_T2_END = 256
_T3_START = 256
_T3_END = 400

_SKETCH_DEPTH = 4
_SKETCH_WIDTH = 90

# Byte offsets
_T1_OFFSET = 0
_T2_OFFSET = _T1_OFFSET + _T1_SIZE * 4  # 64
_T3_OFFSET = _T2_OFFSET + (_T2_END - _T2_START) * 1  # 64 + 240 = 304

# Total check: 304 + 4*90*2 = 304 + 720 = 1024. Correct.

# Hash parameters for Tier 3 sketch
# We need to map local_item (0..143) to 0..89.
# Using linear congruential generator style hashing.
# h(x, r) = (a_r * x + b_r) % 90
_A = [1035493289, 775374203, 1495862053, 2015145641]
_B = [31, 97, 193, 311]

def _sketch_hash(item, row):
    local_item = item - _T3_START
    return (_A[row] * local_item + _B[row]) % _SKETCH_WIDTH

def _read_u32(buf, off):
    return buf[off] | (buf[off+1] << 8) | (buf[off+2] << 16) | (buf[off+3] << 24)

def _write_u32(buf, off, val):
    buf[off] = val & 0xFF
    buf[off+1] = (val >> 8) & 0xFF
    buf[off+2] = (val >> 16) & 0xFF
    buf[off+3] = (val >> 24) & 0xFF

def _read_u16(buf, off):
    return buf[off] | (buf[off+1] << 8)

def _write_u16(buf, off, val):
    buf[off] = val & 0xFF
    buf[off+1] = (val >> 8) & 0xFF

def update(state, item):
    buf = bytearray(state)
    
    if item < _T1_SIZE:
        # Tier 1: uint32 exact count
        off = _T1_OFFSET + item * 4
        v = _read_u32(buf, off)
        if v < 0xFFFFFFFF:
            _write_u32(buf, off, v + 1)
    elif item < _T2_END:
        # Tier 2: uint8 exact count (capped)
        off = _T2_OFFSET + (item - _T2_START)
        v = buf[off]
        if v < 255:
            buf[off] = v + 1
    else:
        # Tier 3: Count-Min Sketch
        for r in range(_SKETCH_DEPTH):
            idx = _sketch_hash(item, r)
            off = _T3_OFFSET + (r * _SKETCH_WIDTH + idx) * 2
            v = _read_u16(buf, off)
            if v < 0xFFFF:
                _write_u16(buf, off, v + 1)
            
    return bytes(buf)

def query(state, item):
    if item < _T1_SIZE:
        off = _T1_OFFSET + item * 4
        return _read_u32(state, off)
    elif item < _T2_END:
        off = _T2_OFFSET + (item - _T2_START)
        return state[off]
    else:
        mn = 0xFFFF
        for r in range(_SKETCH_DEPTH):
            idx = _sketch_hash(item, r)
            off = _T3_OFFSET + (r * _SKETCH_WIDTH + idx) * 2
            v = _read_u16(state, off)
            if v < mn:
                mn = v
        return mn
