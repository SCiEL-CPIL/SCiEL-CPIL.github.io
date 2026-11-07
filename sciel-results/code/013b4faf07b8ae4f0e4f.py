"""
Combined streaming frequency sketch.
Strategy: Use a 1-row Count-Min Sketch with 512 columns (uint16 counters).
Rationale:
- The seed used 4 rows of 64 uint32 counters.
- The parent (score 2.29) used 1 row of 256 uint32 counters.
- The current program (score 30.8) uses 1 row of 512 uint16 counters.
- Increasing columns from 256 to 512 halves the expected collision noise
  (from ~4 to ~2), which should significantly reduce Mean Absolute Error.
- uint16 is sufficient because the maximum possible count in any bucket is 1024,
  which is well below 65535, so no overflow occurs.
- Total memory: 512 * 2 bytes = 1024 bytes, within the limit.

The hash function maps items 0..399 to 0..511 using a strong polynomial mixer
to avoid systematic collisions.
"""

def update(state, item):
    # Convert immutable bytes to mutable bytearray
    b = bytearray(state)
    
    # Hash function: strong polynomial mix mapped to 512 buckets
    # Using the same mixer as the seed but with modulus 512
    idx = ((item * 1035493289 + 31) % 2147483647) % 512
    offset = idx * 2
    
    # Unpack little-endian uint16
    val = int.from_bytes(b[offset:offset+2], 'little')
    
    # Increment
    val += 1
    
    # Pack back
    b[offset:offset+2] = val.to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    # Hash function: same as in update
    idx = ((item * 1035493289 + 31) % 2147483647) % 512
    offset = idx * 2
    
    # Unpack little-endian uint16
    val = int.from_bytes(state[offset:offset+2], 'little')
    
    return val
