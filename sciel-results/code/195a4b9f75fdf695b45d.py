import struct

# EVOLVE-BLOCK-START
def update(state, item):
    # State: 1024 bytes.
    # Layout: 400 uint16 counters (800 bytes) + 224 bytes padding.
    # This allows exact counting for items 0..399.
    
    s = bytearray(state)
    
    # Calculate offset for the item's counter
    # Each counter is 2 bytes (uint16)
    offset = item * 2
    
    # Read current count
    # struct.unpack_from requires a buffer and offset
    count = struct.unpack_from('<H', s, offset)[0]
    
    # Increment
    count += 1
    
    # Check for overflow (should not happen with 1024 items, max count 1024 < 65535)
    if count > 0xFFFF:
        count = 0 # Wrap around, though unlikely
    
    # Write back
    struct.pack_into('<H', s, offset, count)
    
    return bytes(s)

def query(state, item):
    # State: 1024 bytes.
    # Layout: 400 uint16 counters (800 bytes) + 224 bytes padding.
    
    s = bytearray(state)
    
    # Calculate offset
    offset = item * 2
    
    # Read and return count
    count = struct.unpack_from('<H', s, offset)[0]
    
    return count
# EVOLVE-BLOCK-END
