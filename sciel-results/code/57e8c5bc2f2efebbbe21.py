"""Memory-bounded streaming frequency sketch with hybrid exact+CM design."""
import struct
import array

# Constants
NUM_ROWS = 4
NUM_COLS = 64
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
MOD_LARGE = 2147483647

# State layout:
# Bytes 0-399: exact 8-bit counters for items 0-399 (capped at 255)
# Bytes 400-1023: 624 bytes for 4 rows of 64 uint32 CM counters (1024 bytes)
# Total: 400 + 1024 = 1424 > 1024. Need to be more careful.

# Actually, let's reconsider. We have 1024 bytes total.
# Option: Use 400 bytes for exact counters (items 0-399, 1 byte each, capped at 255)
# Remaining 624 bytes for CM sketch. 624 / 4 rows = 156 bytes per row.
# 156 / 4 = 39 uint32 counters per row. That's not a power of 2, but works.
# Alternatively: 384 bytes for exact counters (items 0-383), remaining 640 for CM.
# 640 / 4 = 160 bytes per row, 160/4 = 40 uint32 per row.

# Let's try: 400 exact counters (1 byte each, items 0-399), capped at 255.
# Remaining: 1024 - 400 = 624 bytes for CM.
# 624 bytes = 156 uint32 counters total. With 4 rows, 39 per row.
# This is awkward. Let's use 3 rows of 40 uint32 = 120 uint32 = 480 bytes.
# 400 + 480 = 880 bytes. Remaining 144 bytes unused or for extension.

# Better approach: 
# 400 bytes: exact counters for items 0-399 (1 byte each, 0-255)
# 480 bytes: 3 rows x 40 uint32 CM counters
# 144 bytes: spare (could use for high-bit extension of heavy hitters)

# For Zipf(1.6) on 0..399, items 0-19 are heavy hitters.
# Let's reserve some bytes for high-bit counters of items 0-31 (32 items).
# 32 bytes for high bits (each item can have count up to ~1024, so 2 bits per item = 64 bits = 8 bytes)
# Actually, let's just use the exact counters and cap at 255, and use CM for the rest.

# Final layout:
# Bytes 0-399: exact 8-bit counters for items 0-399
# Bytes 400-879: 3 rows x 40 uint32 CM counters (480 bytes)
# Bytes 880-1023: 144 bytes spare (leave as zero)

def _hash(item, row, num_cols):
    p = PRIMES[row]
    return ((item * p + 31 * row) % MOD_LARGE) % num_cols

def update(state, item):
    state = bytearray(state)
    
    # Update exact counter (bytes 0-399)
    if item < 400:
        if state[item] < 255:
            state[item] += 1
    
    # Update CM sketch (bytes 400-879, 3 rows x 40 uint32)
    num_cols = 40
    for row in range(3):
        idx = _hash(item, row, num_cols)
        offset = 400 + row * num_cols * 4 + idx * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < 0xFFFFFFFF:
            val += 1
        struct.pack_into('<I', state, offset, val)
    
    return bytes(state)

def query(state, item):
    # Get exact count (capped at 255)
    exact = state[item] if item < 400 else 0
    
    # Get CM estimate (3 rows)
    num_cols = 40
    cm_estimates = []
    for row in range(3):
        idx = _hash(item, row, num_cols)
        offset = 400 + row * num_cols * 4 + idx * 4
        val = struct.unpack_from('<I', state, offset)[0]
        cm_estimates.append(val)
    
    cm_min = min(cm_estimates)
    
    # Combine: use max of exact and CM min, but exact is capped at 255
    # If exact is 255, the true count might be higher, so use CM if it's larger
    estimate = max(exact, cm_min)
    
    # Ensure non-negative
    return max(0, estimate)
