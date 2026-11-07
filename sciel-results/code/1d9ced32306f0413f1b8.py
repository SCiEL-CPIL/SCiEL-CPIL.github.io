"""
Exact frequency counter for items 0..399.

Uses 1024 bytes of state.
Allocates 800 bytes (400 items * 2 bytes/item) for exact 16-bit counters.
Remaining 224 bytes are unused (zeroed).

Since the maximum possible frequency in any stream is 1024, and we accumulate
3 streams, the maximum cumulative count is 3072.
Wait, 3 * 1024 = 3072.
A 16-bit unsigned integer can hold up to 65535.
So 16-bit is sufficient for the cumulative count of 3 streams.

If the streams are independent and we query the final state, the estimate is the
sum of counts from all 3 streams.
"""

import struct

def update(state, item):
    """
    Update the state by incrementing the counter for the given item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: bytes (1024 bytes)
    """
    if item < 0 or item > 399:
        # Fallback to do nothing if item is out of bounds, though contract says 0..399
        return state
        
    ba = bytearray(state)
    
    # Calculate offset for 16-bit little-endian integer
    offset = item * 2
    
    # Read current value
    val = ba[offset] | (ba[offset + 1] << 8)
    
    # Increment
    val += 1
    
    # Write back
    ba[offset] = val & 0xFF
    ba[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    """
    Query the estimated frequency for the given item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: nonnegative integer (frequency estimate)
    """
    if item < 0 or item > 399:
        return 0

    # Calculate offset for 16-bit little-endian integer
    offset = item * 2
    
    # Read value
    val = state[offset] | (state[offset + 1] << 8)
    
    return val

# Initialize state with zeros if run directly (though evaluator provides initial state)
if __name__ == "__main__":
    initial_state = bytes(1024)
    # Test
    s = initial_state
    s = update(s, 0)
    s = update(s, 0)
    s = update(s, 399)
    print(query(s, 0)) # Should be 2
    print(query(s, 399)) # Should be 1
    print(len(s)) # Should be 1024
