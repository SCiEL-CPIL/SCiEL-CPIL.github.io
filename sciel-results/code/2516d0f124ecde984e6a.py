"""Domain-adaptive frequency sketch: hot exact counters + widened CM sketch."""

# Precompute hashes for the "cold" items (20..399) for 6 rows x 128 cols
# We use 6 different odd multipliers and offsets to decorrelate rows.
_COLD_BASE = 20
_COLD_COUNT = 380  # items 20..399

# 6 rows, 128 columns each -> 768 counters * 4 bytes = 3072 bytes? No, 1024 limit.
# 1024 bytes total.
# Hot: 20 items * 4 bytes = 80 bytes.
# Remaining: 1024 - 80 = 944 bytes for CM sketch.
# 944 / 4 = 236 counters.
# Let's do 6 rows * 39 cols = 234 counters (936 bytes). Total 80+936 = 1016 bytes.
# Or 5 rows * 47 cols = 235 counters (940 bytes). Total 80+940 = 1020 bytes.
# Let's stick to a clean structure that fits easily.
# Actually, let's use 4-byte counters.
# Hot: 20 * 4 = 80.
# CM: 1024 - 80 = 944 bytes -> 236 uint32s.
# 6 rows * 39 cols = 234. Wastes 2 counters.
# 5 rows * 47 cols = 235. Wastes 1 counter.
# Let's use 6 rows and 39 cols for the cold part.

ROWS = 6
COLS = 39
HOT_SIZE = 20

# Multipliers and offsets for the 6 rows
# Must be such that (item * p + q) % COLS is well distributed for item in 0..399
# Since COLS is small (39), we need good mixing.
P_VALS = [1035493289, 775374203, 1495862053, 2015145641, 1103515245, 1234567891]
Q_VALS = [31, 17, 53, 79, 11, 41]

# Precompute indices for cold items (20..399)
# index_in_state = HOT_SIZE * 4 + (row * COLS + col) * 4
# We will precompute the byte offsets for each cold item and row to speed up update/query
# But update/query must be fast. Precomputing a lookup table for all 400 items x 6 rows is fine.
# LUT[item][row] = byte_offset

_LUT = [[0] * ROWS for _ in range(400)]

for item in range(_COLD_BASE, 400):
    for r in range(ROWS):
        p = P_VALS[r]
        q = Q_VALS[r]
        col = (item * p + q) % COLS
        # Offset in state
        # Hot counters are at 0..79
        # CM counters start at 80
        # Counter index = r * COLS + col
        # Byte offset = 80 + (r * COLS + col) * 4
        _LUT[item][r] = 80 + (r * COLS + col) * 4

# For hot items (0..19), the offset is simply item * 4
for item in range(HOT_SIZE):
    for r in range(ROWS):
        _LUT[item][r] = item * 4  # We only use row 0 effectively for hot, but keep structure

def update(state, item):
    import struct
    state = bytearray(state)
    
    if item < HOT_SIZE:
        # Exact count for hot items
        off = item * 4
        val = struct.unpack_from('<I', state, off)[0]
        val += 1
        struct.pack_into('<I', state, off, val)
    else:
        # Increment CM counters
        for r in range(ROWS):
            off = _LUT[item][r]
            val = struct.unpack_from('<I', state, off)[0]
            val += 1
            struct.pack_into('<I', state, off, val)
            
    return bytes(state)

def query(state, item):
    import struct
    
    if item < HOT_SIZE:
        # Exact count
        off = item * 4
        return float(struct.unpack_from('<I', state, off)[0])
    else:
        # Min over rows
        min_val = float('inf')
        for r in range(ROWS):
            off = _LUT[item][r]
            val = struct.unpack_from('<I', state, off)[0]
            if val < min_val:
                min_val = val
        return float(min_val)
