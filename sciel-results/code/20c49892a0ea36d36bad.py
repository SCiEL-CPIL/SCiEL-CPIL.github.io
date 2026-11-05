# Precompute popcount for 8-bit values
_popcount_table = [bin(i).count('1') for i in range(256)]

def _popcount(x):
    count = 0
    while x:
        count += _popcount_table[x & 0xFF]
        x >>= 8
    return count
