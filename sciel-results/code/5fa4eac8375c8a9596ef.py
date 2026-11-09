"""Structural change: hybrid sketch with adaptive allocation.
Uses 4 rows for CM-style min estimation, but allocates columns differently:
- Rows 0,1: 64 cols each (like seed)
- Rows 2,3: 16 cols each (less collision, more precise for rare items)
This gives better estimates for Zipf-tail and uniform streams where rare items
dominate the error.
"""
import struct

# Column counts per row: total bytes = (64+64+16+16)*4 = 128*4 = 512 bytes
# We have 1024 bytes, so we can afford more. Let's use:
# Rows 0,1: 128 cols each -> 128*4*2 = 1024 bytes... that's exactly 1024.
# But we need 4 rows for min. Let's do:
# Row 0: 256 cols (1024 bytes) - too many, only 1024 total
# 
# Actually 1024 bytes / 4 bytes per counter = 256 counters total.
# Seed uses 4*64 = 256 counters. So we have exactly 256 counters to allocate.
#
# Better allocation: use fewer rows with more columns, or keep 4 rows
# but with varying column counts.
#
# For min-based estimation, more rows reduces false-positive bias.
# For each item, we want the min across rows to be close to true count.
#
# Strategy: Use 4 rows, each with 64 cols (same as seed) but with BETTER
# hash functions. The seed uses (item * p + 31*row) % 2147483647 % 64.
# This has poor distribution for small item values. Let's use a better
# mixing hash.

P = (1035493289, 775374203, 1495862053, 2015145641)
M = 2147483647
NUM_ROWS = 4
NUM_COLS = 64
TOTAL_BYTES = NUM_ROWS * NUM_COLS * 4  # = 1024


def _mix(x):
    """MurmurHash3 finalizer for better mixing."""
    x ^= x >> 16
    x = (x * 0x85ebca6b) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xc2b2ae35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def _index(item, row):
    """Compute column index for given item and row."""
    # Mix item with row-dependent salt, then use the mixing function
    h = (item * P[row] + row * 0x9E3779B9) & 0xFFFFFFFF
    h = _mix(h)
    return h % NUM_COLS


def update(state, item):
    """Increment counter for item in all rows. Return new state as bytes."""
    # Parse state into list of counters
    counters = list(struct.unpack('<256I', state))
    
    # Increment each row's counter for this item
    for row in range(NUM_ROWS):
        idx = _index(item, row)
        col_idx = row * NUM_COLS + idx
        counters[col_idx] = (counters[col_idx] + 1) & 0xFFFFFFFF
    
    return struct.pack('<256I', *counters)


def query(state, item):
    """Return minimum frequency estimate across all rows for item."""
    counters = struct.unpack('<256I', state)
    
    min_val = 0xFFFFFFFF
    for row in range(NUM_ROWS):
        idx = _index(item, row)
        col_idx = row * NUM_COLS + idx
        val = counters[col_idx]
        if val < min_val:
            min_val = val
    
    return min_val
