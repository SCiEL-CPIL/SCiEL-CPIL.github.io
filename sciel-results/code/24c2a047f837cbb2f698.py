"""Memory-bounded streaming frequency sketch: 16 rows x 64 uint8 counters."""

# 16 odd constants for hashing (coprime to 64 for uniform distribution mod 64)
_ROW_CONSTANTS = (
    1035493289, 775374203, 1495862053, 2015145641,
    1103515245, 1234567891, 2002103513, 2654435761,
    3001111111, 3141592653, 2718281829, 1618033989,
    1357246801, 2468135771, 3276813579, 4123456789,
)
_NUM_ROWS = 16
_NUM_COLS = 64


def update(state, item):
    """Increment the counter for item in each of the 16 rows."""
    buf = bytearray(state)
    for row in range(_NUM_ROWS):
        c = _ROW_CONSTANTS[row]
        idx = (item * c + row * 31) % _NUM_COLS
        pos = row * _NUM_COLS + idx
        buf[pos] = (buf[pos] + 1) & 0xFF
    return bytes(buf)


def query(state, item):
    """Return the minimum counter value across all 16 rows for item."""
    min_val = 255
    for row in range(_NUM_ROWS):
        c = _ROW_CONSTANTS[row]
        idx = (item * c + row * 31) % _NUM_COLS
        pos = row * _NUM_COLS + idx
        v = state[pos]
        if v < min_val:
            min_val = v
    return min_val
