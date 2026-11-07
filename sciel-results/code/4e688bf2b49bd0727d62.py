"""
Direct-mapped 16-bit frequency table.
Each of the 400 items has its own 2-byte (uint16) counter.
State layout:
  bytes 0..799:   400 x 2-byte little-endian uint16 counters
  bytes 800..1023: unused padding (all zeros)

This provides exact counts for all items since max possible count is 1024 (fits in uint16).
No collisions, no quantization error, no sketch overestimation.
"""

def update(state, item):
    buf = bytearray(state)
    # Each item gets 2 bytes at offset item * 2
    offset = item * 2
    # Read current 16-bit little-endian value
    val = buf[offset] | (buf[offset + 1] << 8)
    # Increment (max possible is 1024, well within uint16 range of 65535)
    val += 1
    # Write back
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF
    return bytes(buf)


def query(state, item):
    offset = item * 2
    return state[offset] | (state[offset + 1] << 8)
