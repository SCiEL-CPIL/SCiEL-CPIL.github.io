import struct

ROWS = 16
WIDTH = 64
# 16 primes
PRIMES = (
    1035493289, 775374203, 1495862053, 2015145641,
    1103515245, 1234567891, 1357924681, 1478967303,
    1597463007, 1680700013, 1794246733, 1899884863,
    1987654321, 2048362973, 2147483629, 2236962103
)
MOD = 2147483647

def _row_col(item, row):
    p = PRIMES[row]
    return (item * p + 31 * row) % MOD % WIDTH

def update(state, item):
    buf = bytearray(state)
    for row in range(ROWS):
        col = _row_col(item, row)
        idx = row * WIDTH + col
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
    return bytes(buf)

def query(state, item):
    best = 255
    for row in range(ROWS):
        col = _row_col(item, row)
        idx = row * WIDTH + col
        val = state[idx]
        if val < best:
            best = val
    return best
