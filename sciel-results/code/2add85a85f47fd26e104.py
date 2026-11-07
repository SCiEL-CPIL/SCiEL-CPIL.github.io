import struct

NUM_ROWS = 4
NUM_COLS = 128
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)

def _idx(item, row):
    p = PRIMES[row]
    return ((item * p + 31 * row) % 2147483647) % NUM_COLS

def update(state, item):
    counters = list(struct.unpack('<512H', state))
    for r in range(NUM_ROWS):
        idx = _idx(item, r)
        pos = r * NUM_COLS + idx
        counters[pos] = min(counters[pos] + 1, 65535)
    return struct.pack('<512H', *counters)

def query(state, item):
    counters = struct.unpack('<512H', state)
    return min(counters[r * NUM_COLS + _idx(item, r)] for r in range(NUM_ROWS))
