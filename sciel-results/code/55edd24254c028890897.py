"""Memory-bounded streaming frequency sketches: exact sparse map in 1024 bytes using uint16 counters."""
import struct

# EVOLVE-BLOCK-START

def update(state, item):
    """Increment the exact count for 'item' in the 1024-byte state.
    
    State layout: 400 uint16 little-endian counters (800 bytes) at offset 0, 
    each storing the count (capped at 65535, though max possible is 1024)
    for item index 0..399. Remaining 224 bytes are unused padding.
    """
    # Parse the 800-byte count table into a mutable bytearray
    # We only care about the first 800 bytes
    counts = bytearray(state[:800])
    idx = item % 400
    offset = idx * 2
    
    # Read current value
    current = counts[offset] | (counts[offset + 1] << 8)
    
    # Increment, capping at 65535 (though 1024 is the max stream length)
    if current < 65535:
        current += 1
    
    # Write back
    counts[offset] = current & 0xFF
    counts[offset + 1] = (current >> 8) & 0xFF
    
    # Rebuild state: 800 bytes of counts + 224 bytes of zero padding
    new_state = bytes(counts) + b'\x00' * 224
    return new_state

def query(state, item):
    """Return the exact count for 'item'."""
    idx = item % 400
    offset = idx * 2
    return int(state[offset] | (state[offset + 1] << 8))

# EVOLVE-BLOCK-END
