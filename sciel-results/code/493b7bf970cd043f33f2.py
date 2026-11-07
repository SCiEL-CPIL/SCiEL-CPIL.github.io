"""Improved frequency sketch: wider CM with uint16 counters + exact bitset for zero/one detection."""
import struct
import array

# Layout within 1024 bytes:
#   bytes 0..49:   50 bytes = 400-bit membership bitset (bit i set if item i has been seen)
#   bytes 50..509: 460 bytes = 230 uint16 counters? No, let's use a cleaner layout.
#
# Better layout:
#   bytes 0..49:   400-bit bitset (50 bytes)
#   bytes 50..1023: 974 bytes for sketch counters
#
# For the sketch, use 4 rows of 61 uint16 counters = 4*61*2 = 488 bytes.
# That leaves 974-488 = 486 bytes unused. We can use more rows or wider.
#
# Actually, let's use: 6 rows of 64 uint16 counters = 6*64*2 = 768 bytes.
# 50 + 768 = 818 bytes. Fits in 1024.
#
# Or: 8 rows of 60 uint16 = 8*60*2 = 960. 50+960=1010. Fits.
#
# Let's go with 6 rows x 64 uint16 = 768 bytes, starting at offset 50.
# Total used: 50 + 768 = 818 bytes.

ROWS = 6
WIDTH = 64
COUNTER_BYTES = ROWS * WIDTH * 2  # 768
BITSET_BYTES = 50
BASE_OFFSET = BITSET_BYTES  # 50

# Hash primes for each row (use different primes for better distribution)
PRIMES = [1035493289, 775374203, 1495862053, 2015145641, 1234567891, 876543211]
MOD = 2147483647

def _hash_index(item, row):
    """Compute the column index for a given item and row."""
    p = PRIMES[row % len(PRIMES)]
    return ((item * p + 31 * row) % MOD) % WIDTH

def update(state, item):
    """Update the sketch state with a new item."""
    # Work with a mutable copy
    buf = bytearray(state)
    
    # Set the membership bit for this item
    byte_idx = item // 8
    bit_idx = item % 8
    buf[byte_idx] |= (1 << bit_idx)
    
    # Increment the counters in each row
    for row in range(ROWS):
        col = _hash_index(item, row)
        # Each counter is uint16, little-endian
        offset = BASE_OFFSET + row * WIDTH * 2 + col * 2
        # Read current value
        val = buf[offset] | (buf[offset + 1] << 8)
        # Increment (cap at 65535)
        val += 1
        if val > 65535:
            val = 65535
        # Write back
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    """Query the estimated frequency of an item."""
    buf = state
    
    # Check if the item has ever been seen
    byte_idx = item // 8
    bit_idx = item % 8
    if not (buf[byte_idx] & (1 << bit_idx)):
        return 0.0
    
    # Compute the minimum across all rows
    min_val = 65536
    for row in range(ROWS):
        col = _hash_index(item, row)
        offset = BASE_OFFSET + row * WIDTH * 2 + col * 2
        val = buf[offset] | (buf[offset + 1] << 8)
        if val < min_val:
            min_val = val
    
    return float(min_val)
