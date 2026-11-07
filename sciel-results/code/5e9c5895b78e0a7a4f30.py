"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

# We use a hybrid approach:
# 1. First 400 bytes: exact counts for each item (0..399), stored as uint8.
# 2. Remaining 624 bytes: a Count-Min Sketch for overflow handling.
#    The CMS uses 4 rows of 156 uint8 counters (4*156 = 624 bytes).
#    This gives us a wide CMS to reduce collision noise for any overflow.

# CMS configuration
CMS_ROWS = 4
CMS_COLS = 156
CMS_OFFSET = 400  # Start of CMS in the state bytes
MOD = 2147483647
P_VALUES = (1035493289, 775374203, 1495862053, 2015145641)

def update(state, item):
    b = bytearray(state)
    
    # Update exact count
    current_count = b[item]
    if current_count < 255:
        b[item] = current_count + 1
    else:
        # Overflow: increment CMS counters instead
        for row in range(CMS_ROWS):
            p = P_VALUES[row]
            idx = ((item * p + 31 * row) % MOD) % CMS_COLS
            offset = CMS_OFFSET + row * CMS_COLS + idx
            b[offset] = (b[offset] + 1) & 0xFF
            
    return bytes(b)

def query(state, item):
    exact_count = state[item]
    
    if exact_count < 255:
        # No overflow, return exact count
        return exact_count
    else:
        # Overflow occurred, estimate overflow using CMS
        min_val = 255
        for row in range(CMS_ROWS):
            p = P_VALUES[row]
            idx = ((item * p + 31 * row) % MOD) % CMS_COLS
            offset = CMS_OFFSET + row * CMS_COLS + idx
            val = state[offset]
            if val < min_val:
                min_val = val
        
        # The CMS counts only the overflow (increments after 255)
        # So total estimate is 255 + min_val
        return 255 + min_val
# EVOLVE-BLOCK-END
