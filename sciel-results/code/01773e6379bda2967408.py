import struct

# EVOLVE-BLOCK-START
# Improved Count-Min Sketch with 8 rows of 64 16-bit counters.
# Total: 8 * 64 * 2 = 1024 bytes (exactly fits the state budget).
# Using uint16 counters allows 8 rows instead of 4 (with uint32),
# reducing collision noise and improving accuracy for low-frequency items.

PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2285881789,
    2473401169,
    2600822929,
    2807566739
]
MOD = 2147483647
WIDTH = 64
DEPTH = 8
COUNTER_SIZE = 2  # bytes per counter (uint16)

def update(state, item):
    b = bytearray(state)
    for i in range(DEPTH):
        p = PRIMES[i]
        idx = ((item * p + 31 * i) % MOD) % WIDTH
        offset = i * WIDTH * COUNTER_SIZE + idx * COUNTER_SIZE
        val = struct.unpack_from('<H', b, offset)[0]
        val += 1
        if val > 65535:
            val = 65535
        struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    min_val = 65535
    for i in range(DEPTH):
        p = PRIMES[i]
        idx = ((item * p + 31 * i) % MOD) % WIDTH
        offset = i * WIDTH * COUNTER_SIZE + idx * COUNTER_SIZE
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END
