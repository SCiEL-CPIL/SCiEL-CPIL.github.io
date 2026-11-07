"""
Optimized streaming frequency sketch.
Strategy: 2-row Count-Min Sketch with 512 uint16 counters per row.
- Row 0: bytes [0:1024), hash h1
- Row 1: bytes [1024:2048) — wait, we only have 1024 bytes total.

Actually: 1024 bytes / 2 rows = 512 bytes per row = 256 uint16 counters per row.
So: 2 rows × 256 uint16 counters.
- Row 0: bytes [0:512), 256 counters
- Row 1: bytes [512:1024), 256 counters
Query returns min of the two rows' estimates.

Expected collision noise per row: N/W = 1024/256 = 4.
Min of 2 independent rows reduces expected overestimation significantly.
"""

def _hash(item, row):
    """Independent hash functions for each row."""
    if row == 0:
        # Use a strong multiplier
        return ((item * 1035493289 + 31) % 2147483647) % 256
    else:
        # Different multiplier and offset for independence
        return ((item * 775374203 + 107) % 2147483647) % 256

def update(state, item):
    b = bytearray(state)
    
    for row in range(2):
        idx = _hash(item, row)
        offset = row * 512 + idx * 2  # 512 bytes per row, 2 bytes per counter
        
        val = int.from_bytes(b[offset:offset+2], 'little')
        val += 1
        b[offset:offset+2] = val.to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    min_val = 65535  # max uint16
    
    for row in range(2):
        idx = _hash(item, row)
        offset = row * 512 + idx * 2
        
        val = int.from_bytes(state[offset:offset+2], 'little')
        if val < min_val:
            min_val = val
    
    return min_val
