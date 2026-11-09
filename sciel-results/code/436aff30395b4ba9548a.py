import struct

# 4 rows x 256 columns of uint8 counters = 1024 bytes
W = 256
D = 4
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
PRIME = 2147483647

# Precompute all indices: _IDX[item] = (idx0, idx1, idx2, idx3)
_IDX = []
for item in range(400):
    row_indices = []
    for r in range(D):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % PRIME) % W
        row_indices.append(idx)
    _IDX.append(tuple(row_indices))

def update(state, item):
    buf = bytearray(state)
    i0, i1, i2, i3 = _IDX[item]
    
    v = buf[i0]
    buf[i0] = v + 1 if v < 255 else 255
    
    v = buf[256 + i1]
    buf[256 + i1] = v + 1 if v < 255 else 255
    
    v = buf[512 + i2]
    buf[512 + i2] = v + 1 if v < 255 else 255
    
    v = buf[768 + i3]
    buf[768 + i3] = v + 1 if v < 255 else 255
    
    return bytes(buf)

def query(state, item):
    i0, i1, i2, i3 = _IDX[item]
    
    v0 = state[i0]
    v1 = state[256 + i1]
    v2 = state[512 + i2]
    v3 = state[768 + i3]
    
    m = v0
    if v1 < m:
        m = v1
    if v2 < m:
        m = v2
    if v3 < m:
        m = v3
    
    return m
