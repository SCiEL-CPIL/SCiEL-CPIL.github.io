"""
Hybrid exact-counting sketch.
Uses the first 400 bytes of the state as a direct address table for items 0..399.
Each byte holds the exact count for that item, capped at 255.
This eliminates collision error entirely for counts < 256.
Remaining bytes are unused but preserved to maintain state size.
"""

# EVOLVE-BLOCK-START
def update(state, item):
    """
    Update the state by incrementing the count for the given item.
    state: bytes of length 1024.
    item: int in 0..399.
    returns: new state bytes of length 1024.
    """
    # Convert to bytearray for mutation
    ba = bytearray(state)
    
    # Increment count, capping at 255 to avoid overflow
    if ba[item] < 255:
        ba[item] += 1
    
    return bytes(ba)

def query(state, item):
    """
    Query the estimated frequency for the given item.
    state: bytes of length 1024.
    item: int in 0..399.
    returns: int frequency estimate.
    """
    # Return the exact count stored in the direct address table
    return int(state[item])
# EVOLVE-BLOCK-END
