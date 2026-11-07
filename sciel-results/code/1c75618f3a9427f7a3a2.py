"""Memory-bounded streaming frequency sketch: hybrid exact + CM sketch."""
import struct

# Layout:
# Bytes 0..19:   20 x uint8 exact counters for items 0-19
# Bytes 20..269: 250 x uint8 exact counters for items 150-399
# Bytes 270..1023: 754 bytes for CM sketch covering items 20-149 (130 items)
#   Use 13 rows x 58 cols x uint8 = 754 bytes
# Total: 20 + 250 + 754 = 1024 bytes

_EXACT_LOW_END = 20
_EXACT_HIGH_START = 270
_SKETCH_OFFSET = 270
_SKETCH_ROWS = 13
_SKETCH_COLS = 58

# Precompute hash indices for items 20-149 in each row
# item_to_col[row][item] = column index for that item in that row
def _compute_hash_table():
    """Precompute all hash indices for the sketch portion."""
    table = []
    for row in range(_SKETCH_ROWS):
        row_data = {}
        for item in range(20, 150):
            # Use a simple hash: (item * prime + row * salt) % cols
            # We need good distribution for 130 items into 58 columns
            # Use multiple primes for different rows
            p = (1035493289 * (row + 1)) % 2147483647
            col = (item * p + 31 * row) % _SKETCH_COLS
            row_data[item] = col
        table.append(row_data)
    return table

_HASH_TABLE = _compute_hash_table()

def update(state, item):
    """Update the sketch state with a new item."""
    ba = bytearray(state)
    
    if item < 20:
        # Exact counter for items 0-19
        if ba[item] < 255:
            ba[item] += 1
    elif item >= 150:
        # Exact counter for items 150-399, offset by 20
        idx = 20 + (item - 150)
        if ba[idx] < 255:
            ba[idx] += 1
    else:
        # Items 20-149: use CM sketch
        for row in range(_SKETCH_ROWS):
            col = _HASH_TABLE[row][item]
            offset = _SKETCH_OFFSET + row * _SKETCH_COLS + col
            if ba[offset] < 255:
                ba[offset] += 1
    
    return bytes(ba)

def query(state, item):
    """Query the estimated frequency of an item."""
    if item < 20:
        return float(state[item])
    elif item >= 150:
        idx = 20 + (item - 150)
        return float(state[idx])
    else:
        # Items 20-149: use CM sketch, return minimum across rows
        min_val = 255
        for row in range(_SKETCH_ROWS):
            col = _HASH_TABLE[row][item]
            offset = _SKETCH_OFFSET + row * _SKETCH_COLS + col
            val = state[offset]
            if val < min_val:
                min_val = val
        return float(min_val)
