"""
Exact Counting Strategy:
Allocates 2 bytes (uint16) per item for all 400 possible items.
Total usage: 400 * 2 = 800 bytes.
Remaining 224 bytes are unused (kept as zero).
Max possible count per item is 3 * 1024 = 3072, which fits in uint16.
This yields 0 error.
"""

_NUM_ITEMS = 400
_BYTES_PER_ITEM = 2
_TOTAL_USED = _NUM_ITEMS * _BYTES_PER_ITEM  # 800
_TOTAL_STATE_SIZE = 1024

def update(state, item):
    buf = bytearray(state)
    if item < 0 or item >= _NUM_ITEMS:
        # Item out of range, do nothing (or handle gracefully)
        return bytes(buf)
    
    off = item * _BYTES_PER_ITEM
    # Read current uint16
    # Little-endian assumed based on seed "little-endian uint32"
    val = buf[off] | (buf[off + 1] << 8)
    
    # Increment
    val += 1
    
    # Write back
    buf[off] = val & 0xFF
    buf[off + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    if item < 0 or item >= _NUM_ITEMS:
        return 0
    off = item * _BYTES_PER_ITEM
    val = state[off] | (state[off + 1] << 8)
    return val
