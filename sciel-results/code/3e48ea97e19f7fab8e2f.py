import struct
import array

def _mix(x):
    # 32-bit mixing
    x = (x ^ (x >> 16)) * 0x45d9f3b
    x = x & 0xFFFFFFFF
    x = (x ^ (x >> 16)) * 0x45d9f3b
    x = x & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return x

# Pre-compute constants
# We want 4 rows.
# p values should be distinct large odd numbers.
P = (1035493289, 775374203, 1495862053, 2015145641)
OFF = (0, 31, 62, 93)

def update(state, item):
    # Use array for faster access
    counters = array.array('H')
    counters.frombytes(state)
    
    for r in range(4):
        # Mix the hash
        h = _mix(item * P[r] + OFF[r])
        idx = h & 127
        offset = r * 128 + idx
        counters[offset] += 1
        
    return counters.tobytes()

def query(state, item):
    counters = array.array('H')
    counters.frombytes(state)
    
    min_val = 65535
    for r in range(4):
        h = _mix(item * P[r] + OFF[r])
        idx = h & 127
        offset = r * 128 + idx
        val = counters[offset]
        if val < min_val:
            min_val = val
            
    return min_val
