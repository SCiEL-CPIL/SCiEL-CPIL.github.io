"""Exact Frequency Table for items 0..399.

Layout: 1024 bytes total.
- First 800 bytes: 400 counters x 2 bytes each (16-bit unsigned integers).
  Index i corresponds to item i.
  Max count is 1024, which fits in 16 bits.
- Remaining 224 bytes: Padding (zeros).

This provides exact frequency counts, resulting in 0 error.
"""

def update(state, item):
    ba = bytearray(state)
    offset = item * 2
    
    # Read 16-bit little-endian integer
    val = ba[offset] | (ba[offset + 1] << 8)
    
    # Increment
    val += 1
    
    # Write back
    ba[offset] = val & 0xFF
    ba[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    ba = state
    offset = item * 2
    
    val = ba[offset] | (ba[offset + 1] << 8)
    
    return val
