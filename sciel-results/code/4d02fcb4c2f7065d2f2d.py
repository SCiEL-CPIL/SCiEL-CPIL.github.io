"""
Structural Change: Full 1024-byte allocation split into 256 exact counters and 
a 32-row x 32-col Count-Min sketch.

Rationale:
The previous strategy used 256 exact counters and a 12-row x 64-col sketch.
This new strategy keeps the 256 exact counters (covering items 0-255, which 
include the heavy hitters from Zipf and Stream 3's first half) but reallocates 
the remaining 768 bytes into a 32-row x 32-col Count-Min sketch.

Why 32 rows?
The score is dominated by the error on items 256-399 (144 items). 
In a CM sketch, the error is the minimum over rows of (true_count + collision_noise).
Increasing the number of rows decreases the probability that the minimum row 
suffers from a collision. 32 rows provide a much lower probability of the 
minimum being inflated compared to 12 rows, even though the column width 
decreases from 64 to 32. 
Given that the items 256-399 are relatively sparse (especially in Stream 1 
uniform and Stream 3 second half), the collision rate in 32 columns is 
manageable, and the benefit of 32 independent estimates outweighs the 
increased collision probability per row.

Layout:
- Bytes 0-255: 256 uint8 exact counters for items 0-255.
- Bytes 256-1023: 32 rows x 32 columns of uint8 counters for items 256-399.
"""
# EVOLVE-BLOCK-START

_NUM_EXACT = 256
_NUM_CM_ROWS = 32
_NUM_CM_COLS = 32
_CM_OFFSET = _NUM_EXACT

# 32 distinct primes for hashing in the CM sketch part
# We need good distribution for items 256-399
# Using a mix of large primes and some smaller ones for variety
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    1919191911,
    16807,
    48271,
    69621,
    28174,
    97531,
    131071,
    262147,
    524287,
    1048573,
    2097151,
    4194301,
    8388607,
    16777213,
    33554429,
    67108853,
    134217727,
    268435447,
    536870909,
    1073741821,
    2147483645,
    314159265,
    271828182,
    161803398,
    123456789,
    987654321,
    456789123,
]
_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    if item < _NUM_EXACT:
        # Exact counting for items 0-255
        val = buf[item]
        if val < 255:
            val += 1
        buf[item] = val
    else:
        # CM sketch for items 256-399
        for row in range(_NUM_CM_ROWS):
            p = _PRIMES[row]
            idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
            offset = _CM_OFFSET + row * _NUM_CM_COLS + idx
            val = buf[offset]
            if val < 255:
                val += 1
            buf[offset] = val
    return bytes(buf)


def query(state, item):
    if item < _NUM_EXACT:
        # Exact count for items 0-255
        return state[item]
    else:
        # CM sketch minimum for items 256-399
        best = 255
        for row in range(_NUM_CM_ROWS):
            p = _PRIMES[row]
            idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
            val = state[_CM_OFFSET + row * _NUM_CM_COLS + idx]
            if val < best:
                best = val
        return best

# EVOLVE-BLOCK-END
