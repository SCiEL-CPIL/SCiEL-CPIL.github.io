import struct

def update(state, item):
    # State is 1024 bytes.
    # We use the first 800 bytes for 400 uint16 counters.
    # item is in 0..399.
    
    # Ensure state is bytes
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
        
    # Convert to bytearray for mutation
    buf = bytearray(state)
    
    # Calculate offset
    offset = item * 2
    
    # Read current count
    current = struct.unpack_from('<H', buf, offset)[0]
    
    # Increment
    new_count = current + 1
    
    # Write back
    struct.pack_into('<H', buf, offset, new_count)
    
    return bytes(buf)

def query(state, item):
    # State is 1024 bytes.
    # item is in 0..399.
    
    # Calculate offset
    offset = item * 2
    
    # Read count
    count = struct.unpack_from('<H', state, offset)[0]
    
    return count
