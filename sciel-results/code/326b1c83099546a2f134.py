"""
Pure Count-Min Sketch with 6 rows x 42 columns.
State: 1008 bytes of little-endian uint32 counters.
6 rows, 42 columns per row.
Query: minimum across all 6 rows.
"""
import struct

# Row parameters
ROWS = 6
COLS = 42
TOTAL_COUNTERS = ROWS * COLS  # 252 counters * 4 bytes = 1008 bytes

# Primes for hashing
PRIMES = [1035493289, 775374203, 1495862053, 2015145641, 165411643, 1063638847]
# Secondary constants for each row to add diversity
Q_CONSTS = [1, 2, 3, 4, 5, 6]

def update(state, item):
    """
    Update the CMS state with one more occurrence of item.
    State is 1024 bytes; first 1008 bytes are the sketch counters.
    """
    new_state = bytearray(state)
    for row in range(ROWS):
        p = PRIMES[row]
        q = Q_CONSTS[row]
        # Hash: ((item * p + 31 * row) % 2147483647) % COLS
        idx = ((item * p + 31 * row + q * 997) % 2147483647) % COLS
        offset = (row * COLS + idx) * 4
        val = struct.unpack_from('<I', new_state, offset)[0]
        struct.pack_into('<I', new_state, offset, val + 1)
    return bytes(new_state)

def query(state, item):
    """
    Query the CMS state for the frequency estimate of item.
    Returns the minimum across all rows.
    """
    min_val = 2**32 - 1
    for row in range(ROWS):
        p = PRIMES[row]
        q = Q_CONSTS[row]
        idx = ((item * p + 31 * row + q * 997) % 2147483647) % COLS
        offset = (row * COLS + idx) * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_val:
            min_val = val
    return float(min_val)
