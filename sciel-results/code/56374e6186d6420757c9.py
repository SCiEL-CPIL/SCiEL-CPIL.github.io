"""
Hybrid Adaptive Sketch:
- Direct Address Table (DAT) for items 0-63 (128 bytes).
- Count-Min Sketch for items 64-399 (896 bytes).
- Total state size: 1024 bytes.
"""

# Configuration
_DAT_SIZE = 64  # Items 0-63 stored exactly
_DAT_BYTES = _DAT_SIZE * 2  # 128 bytes

_CM_OFFSET = _DAT_BYTES  # Start of CM sketch in state
_CM_DEPTH = 2
_CM_WIDTH = 224  # 2 * 224 * 2 = 896 bytes
_CM_BYTES = _CM_DEPTH * _CM_WIDTH * 2

# Hash parameters for CM sketch
# Using primes from the seed to ensure good distribution
_A = [1035493289, 775374203]
_B = [0, 129]

def _read_uint16(buf, offset):
    return buf[offset] | (buf[offset + 1] << 8)

def _write_uint16(buf, offset, val):
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF

def update(state, item):
    buf = bytearray(state)
    
    if item < _DAT_SIZE:
        # Direct Address Table update
        offset = item * 2
        val = _read_uint16(buf, offset)
        val += 1
        _write_uint16(buf, offset, val)
    else:
        # Count-Min Sketch update
        for i in range(_CM_DEPTH):
            # Map item to index in [0, _CM_WIDTH)
            # Use a simple hash: (a * item + b) % width
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            # Offset in state: CM_OFFSET + row * (width * 2) + idx * 2
            byte_offset = _CM_OFFSET + i * (_CM_WIDTH * 2) + idx * 2
            val = _read_uint16(buf, byte_offset)
            val += 1
            _write_uint16(buf, byte_offset, val)
            
    return bytes(buf)

def query(state, item):
    buf = state
    
    if item < _DAT_SIZE:
        # Return exact count from DAT
        offset = item * 2
        return _read_uint16(buf, offset)
    else:
        # Return min of CM counters
        min_val = 65535
        for i in range(_CM_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            byte_offset = _CM_OFFSET + i * (_CM_WIDTH * 2) + idx * 2
            val = _read_uint16(buf, byte_offset)
            if val < min_val:
                min_val = val
        return min_val
