"""
Optimized Count-Min Sketch: 4 rows × 128 uint16 counters.

Structural change from the current program (2×256):
- Switch to 4 rows × 128 columns (uint16) = 1024 bytes.
- This increases depth (min-of-4) while keeping width at 128.
- The key insight: with 400 items and 128 slots, expected load is ~3.1 per slot.
  With 4 independent rows, the probability that ALL 4 rows have collisions
  inflating the count is much lower than with 2 rows.
- The min-of-4 provides tighter upper bounds on the true count.

Hash optimization:
- Precompute all 400×4 indices at module load time (fast, no per-call overhead).
- Use distinct large prime multipliers for good mixing.
- Use int.from_bytes/to_bytes for faster serialization than struct in tight loops.

This is a structural change: different depth/width trade-off (4×128 vs 2×256)
which changes the fundamental error characteristics of the sketch.
"""

import struct

_D = 4
_W = 128

# Verify: _D * _W * 2 = 4 * 128 * 2 = 1024 bytes ✓

# Hash parameters - large primes for good distribution
_PARAMS = [
    (1035493289, 31),
    (775374203, 67),
    (1495862053, 97),
    (2015145641, 131),
]

_M = 2147483647  # 2^31 - 1

# Precompute all indices for items 0..399 across all 4 rows
# _PRECOMP[row][item] = column index
_PRECOMP = []
for row in range(_D):
    a, b = _PARAMS[row]
    row_indices = [0] * 400
    for item in range(400):
        row_indices[item] = (a * item + b) % _M % _W
    _PRECOMP.append(row_indices)


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    
    for i in range(_D):
        idx = _PRECOMP[i][item]
        byte_offset = i * (_W * 2) + idx * 2
        
        current = int.from_bytes(buf[byte_offset:byte_offset+2], 'little')
        new_val = current + 1
        if new_val > 65535:
            new_val = 65535
        buf[byte_offset:byte_offset+2] = new_val.to_bytes(2, 'little')
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    min_count = 65536
    
    for i in range(_D):
        idx = _PRECOMP[i][item]
        byte_offset = i * (_W * 2) + idx * 2
        val = int.from_bytes(state[byte_offset:byte_offset+2], 'little')
        if val < min_count:
            min_count = val
    
    return min_count
