import struct

def update(state, item):
    # state is bytes of length 1024
    # We use the first 800 bytes for 400 uint16 counters
    # item is 0..399
    
    # Convert to bytearray for mutability
    ba = bytearray(state)
    
    # Calculate offset
    offset = item * 2
    
    # Read current value
    # Little-endian uint16
    current_val = ba[offset] | (ba[offset + 1] << 8)
    
    # Increment
    new_val = current_val + 1
    
    # Write back
    ba[offset] = new_val & 0xFF
    ba[offset + 1] = (new_val >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    # state is bytes of length 1024
    # item is 0..399
    
    offset = item * 2
    
    # Read value
    # Little-endian uint16
    val = state[offset] | (state[offset + 1] << 8)
    
    return val
