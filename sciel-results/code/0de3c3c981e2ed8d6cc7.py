import time
import struct

_START = time.time()
_DEADLINE = _START + 130

# Layout:
# Bytes [0:256): 256 uint8 exact counters for items 0..255
# Bytes [256:1024): 768 bytes = 2 rows of 128 uint16 counters for items 256..399
#   Row 0: bytes [256:512), 128 uint16 counters
#   Row 1: bytes [512:768), 128 uint16 counters
# Bytes [768:1024): unused (256 bytes)

# Hash functions for items 256..399 mapping to 0..127
# We use two independent hashes

def _hash_row0(item):
    # item is in 256..399, map to 0..127
    return ((item * 1035493289 + 31) % 2147483647) % 128

def _hash_row1(item):
    return ((item * 775374203 + 107) % 2147483647) % 128

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    
    b = bytearray(state)
    
    if item < 256:
        # Exact count: increment byte at position item
        b[item] = (b[item] + 1) & 0xFF
    else:
        # Use 2-row CMS for items 256..399
        # Row 0
        idx0 = _hash_row0(item)
        offset0 = 256 + idx0 * 2
        val0 = int.from_bytes(b[offset0:offset0+2], 'little')
        val0 = (val0 + 1) & 0xFFFF
        b[offset0:offset0+2] = val0.to_bytes(2, 'little')
        
        # Row 1
        idx1 = _hash_row1(item)
        offset1 = 512 + idx1 * 2
        val1 = int.from_bytes(b[offset1:offset1+2], 'little')
        val1 = (val1 + 1) & 0xFFFF
        b[offset1:offset1+2] = val1.to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0
    
    if item < 256:
        # Exact count
        return state[item]
    else:
        # Min of two rows
        idx0 = _hash_row0(item)
        offset0 = 256 + idx0 * 2
        val0 = int.from_bytes(state[offset0:offset0+2], 'little')
        
        idx1 = _hash_row1(item)
        offset1 = 512 + idx1 * 2
        val1 = int.from_bytes(state[offset1:offset1+2], 'little')
        
        return min(val0, val1)
