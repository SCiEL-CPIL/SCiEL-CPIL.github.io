"""
Exact frequency counting for items 0..399 using 16-bit unsigned integers.
State is 1024 bytes. First 800 bytes store 400 counters (2 bytes each).
Remaining 224 bytes are unused (kept as zero).
"""
# EVOLVE-BLOCK-START
def update(state, item):
    # Convert state to mutable bytearray
    buf = bytearray(state)
    
    # Calculate offset for the item's counter
    # Each item uses 2 bytes (uint16 LE)
    offset = item * 2
    
    # Read current value
    # Using int.from_bytes is fast and safe
    current_val = int.from_bytes(buf[offset:offset+2], byteorder='little')
    
    # Increment
    new_val = current_val + 1
    
    # Write back
    buf[offset:offset+2] = new_val.to_bytes(2, byteorder='little')
    
    # Return as bytes
    return bytes(buf)

def query(state, item):
    # Calculate offset
    offset = item * 2
    
    # Read value
    val = int.from_bytes(state[offset:offset+2], byteorder='little')
    
    return val
# EVOLVE-BLOCK-END
