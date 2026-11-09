"""
Optimized frequency sketch using exact dense counters.
Since the item range is 0..399 (400 items) and the max count per stream is 1024,
we can store exact counts for all items using 2 bytes (uint16) per item.
Total memory required: 400 * 2 = 800 bytes, which is within the 1024 byte limit.
This yields zero error for all queries.
"""

def update(state, item):
    # Convert immutable bytes to mutable bytearray
    ba = bytearray(state)
    
    # Calculate offset for the item's count (2 bytes per item)
    idx = item * 2
    
    # Read current count
    current_count = int.from_bytes(ba[idx:idx+2], 'little')
    
    # Increment
    new_count = current_count + 1
    
    # Write back
    ba[idx:idx+2] = new_count.to_bytes(2, 'little')
    
    return bytes(ba)

def query(state, item):
    # Calculate offset
    idx = item * 2
    
    # Read and return count
    return int.from_bytes(state[idx:idx+2], 'little')
