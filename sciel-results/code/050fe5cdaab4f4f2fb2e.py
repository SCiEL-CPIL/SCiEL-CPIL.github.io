"""
Optimized streaming frequency sketch: 4-row Count-Min Sketch with 128 uint16 counters per row.
Total memory: 4 rows * 128 cols * 2 bytes/col = 1024 bytes.
Rationale: 
- The seed used 4x64 uint32. 
- The previous best used 1x512 uint16.
- This configuration (4x128 uint16) doubles the column width of the seed while maintaining the 4-row variance reduction.
- The min operation across 4 rows significantly reduces the impact of collisions compared to a single row, 
  even though the per-row collision rate is higher than the 1x512 scheme.
- uint16 is sufficient as the maximum possible frequency is well below 65535.
"""

import struct

# Hash parameters for 4 rows
# Using distinct large primes for good mixing
P = [1035493289, 775374203, 1495862053, 2015145641]
C = 128  # Number of columns per row
MOD = 2147483647  # Mersenne prime for mixing

def _get_indices(item):
    """Calculate the 4 indices for the given item."""
    indices = []
    for i in range(4):
        # Mix item with row-specific prime and offset
        # (item * P[i] + 31 * (i+1)) % MOD
        h = (item * P[i] + 31 * (i + 1)) % MOD
        idx = h % C
        indices.append(idx)
    return indices

def update(state, item):
    b = bytearray(state)
    indices = _get_indices(item)
    
    for i, idx in enumerate(indices):
        # Each row starts at i * 128 * 2 bytes
        offset = i * 256 + idx * 2
        # Unpack little-endian uint16
        val = int.from_bytes(b[offset:offset+2], 'little')
        val += 1
        # Pack back
        b[offset:offset+2] = val.to_bytes(2, 'little')
        
    return bytes(b)

def query(state, item):
    indices = _get_indices(item)
    min_val = 65535  # Max uint16
    
    for i, idx in enumerate(indices):
        offset = i * 256 + idx * 2
        val = int.from_bytes(state[offset:offset+2], 'little')
        if val < min_val:
            min_val = val
            
    return min_val
