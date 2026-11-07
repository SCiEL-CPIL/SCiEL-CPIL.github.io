import struct

def update(state, item):
    # state is bytes, length 1024
    # item is int 0..399
    # We use 2 bytes per item, starting at index 0
    # Offset = item * 2
    offset = item * 2
    
    # Convert to bytearray for modification
    buf = bytearray(state)
    
    # Read current value (little-endian uint16)
    # struct.unpack_from('<H', buf, offset)[0]
    # Manual read for speed:
    val = buf[offset] | (buf[offset + 1] << 8)
    
    # Increment
    val += 1
    
    # Write back
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    offset = item * 2
    # Read little-endian uint16
    val = state[offset] | (state[offset + 1] << 8)
    return float(val)
