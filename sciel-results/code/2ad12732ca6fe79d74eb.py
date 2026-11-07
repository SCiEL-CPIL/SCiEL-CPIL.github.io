"""
Optimized streaming frequency sketch: 2-row Count-Min Sketch with nonlinear hashing.
Strategy: 2 rows x 256 uint16 counters = 1024 bytes.
Query returns min of the two row estimates.
Uses multiply-xor-shift hashing for better collision distribution.
"""

def _hash0(item):
    # Nonlinear hash: multiply by odd constant, xor-shift, mod 256
    x = (item + 1) * 0x9E3779B9
    x = ((x >> 16) ^ x) * 0x85EBCA6B
    x = ((x >> 13) ^ x) * 0xC2B2AE35
    x = (x >> 16) ^ x
    return x & 0xFF

def _hash1(item):
    # Different nonlinear hash for row 1
    x = (item + 1) * 0xC2B2AE35
    x = ((x >> 15) ^ x) * 0x9E3779B9
    x = ((x >> 13) ^ x) * 0x85EBCA6B
    x = (x >> 16) ^ x
    return x & 0xFF

def update(state, item):
    b = bytearray(state)
    
    # Row 0
    idx0 = _hash0(item)
    off0 = idx0 * 2
    val0 = int.from_bytes(b[off0:off0+2], 'little')
    val0 += 1
    b[off0:off0+2] = val0.to_bytes(2, 'little')
    
    # Row 1
    idx1 = _hash1(item)
    off1 = 512 + idx1 * 2
    val1 = int.from_bytes(b[off1:off1+2], 'little')
    val1 += 1
    b[off1:off1+2] = val1.to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    # Row 0
    idx0 = _hash0(item)
    off0 = idx0 * 2
    val0 = int.from_bytes(state[off0:off0+2], 'little')
    
    # Row 1
    idx1 = _hash1(item)
    off1 = 512 + idx1 * 2
    val1 = int.from_bytes(state[off1:off1+2], 'little')
    
    return min(val0, val1)
