"""Improved streaming frequency sketch using sparse exact counting in 1024 bytes."""
# EVOLVE-BLOCK-START
import struct

# Precompute the index for each item (0..399) in the 1024-byte state.
# We need 400 entries, each 1 byte (counts 0-255).
# State layout: first 400 bytes are the count table, remaining 624 bytes are unused (zero).
# Index for item i: simply i (since items are 0..399, we can use direct indexing)
# But we need to ensure we stay within 1024 bytes. 400 bytes is well within 1024.

def update(state, item):
    """
    Update the state by incrementing the count for the given item.
    state: bytes of length 1024
    item: int in 0..399
    returns: new state as bytes
    """
    # Convert to bytearray for mutability
    ba = bytearray(state)
    
    # Direct index: item maps to position item in the first 400 bytes
    idx = item  # 0..399
    
    # Increment count, capping at 255
    current = ba[idx]
    if current < 255:
        ba[idx] = current + 1
    # If already 255, keep at 255 (saturating)
    
    return bytes(ba)

def query(state, item):
    """
    Query the estimated frequency for the given item.
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate (int)
    """
    idx = item  # 0..399
    return state[idx]
# EVOLVE-BLOCK-END
