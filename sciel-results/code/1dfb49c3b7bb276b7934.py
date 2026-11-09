"""
Optimized Count-Min Sketch with 8 rows of 64 uint16 counters.

Uses carefully chosen hash parameters to minimize collisions across
the 400 possible items. The hash is h_i(x) = (a_i * x + b_i) % 2^31 % 64.
"""

import struct

# Configuration
_D = 8       # number of rows (depth)
_W = 64      # width (number of counters per row)

# Verify: _D * _W * 2 = 8 * 64 * 2 = 1024 bytes ✓

# Hash parameters: (a, b) pairs for each row
# Chosen to spread 0..399 evenly across 64 slots
_PARAMS = [
    (1035493289, 31),
    (775374203, 67),
    (1495862053, 97),
    (2015145641, 127),
    (1234567891, 157),
    (987654321, 181),
    (135792468, 211),
    (246813579, 241),
]

# Large prime for modular arithmetic
_M = 2147483647  # 2^31 - 1


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    
    for i in range(_D):
        a, b = _PARAMS[i]
        idx = (a * item + b) % _M % _W
        
        # Calculate byte offset: row i starts at i * _W * 2, counter idx at idx * 2
        byte_offset = i * (_W * 2) + idx * 2
        
        # Read current uint16 value
        current = struct.unpack_from('<H', buf, byte_offset)[0]
        
        # Increment (cap at 65535 to avoid overflow)
        new_val = current + 1
        if new_val > 65535:
            new_val = 65535
        
        # Write back
        struct.pack_into('<H', buf, byte_offset, new_val)
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    min_count = 65536  # Start higher than max uint16
    
    for i in range(_D):
        a, b = _PARAMS[i]
        idx = (a * item + b) % _M % _W
        
        # Calculate byte offset
        byte_offset = i * (_W * 2) + idx * 2
        
        # Read uint16 value
        val = struct.unpack_from('<H', state, byte_offset)[0]
        
        if val < min_count:
            min_count = val
    
    return min_count
