import struct

def update(state, item):
    # Create a mutable copy of the state
    buf = bytearray(state)
    
    # Calculate the offset for the item's counter
    # Each counter is 2 bytes (uint16)
    offset = item * 2
    
    # Unpack the current count
    current_count = struct.unpack_from('<H', buf, offset)[0]
    
    # Increment
    new_count = current_count + 1
    
    # Pack the new count back
    struct.pack_into('<H', buf, offset, new_count)
    
    return bytes(buf)

def query(state, item):
    offset = item * 2
    count = struct.unpack_from('<H', state, offset)[0]
    return count
