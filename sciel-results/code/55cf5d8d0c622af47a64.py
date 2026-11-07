"""Memory-bounded streaming frequency sketches - optimized single-row hash table."""

# EVOLVE-BLOCK-START

def update(state, item):
    """Increment the counter for item in the 256-bucket hash table."""
    b = bytearray(state)
    # Golden ratio multiplicative hash for good distribution
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 24) & 0xFF
    offset = idx * 4
    val = int.from_bytes(b[offset:offset+4], 'little')
    val = (val + 1) & 0xFFFFFFFF
    b[offset:offset+4] = val.to_bytes(4, 'little')
    return bytes(b)

def query(state, item):
    """Return the frequency estimate for item from the hash table."""
    h = (item * 2654435761) & 0xFFFFFFFF
    idx = (h >> 24) & 0xFF
    offset = idx * 4
    val = int.from_bytes(state[offset:offset+4], 'little')
    return float(val)

# EVOLVE-BLOCK-END
