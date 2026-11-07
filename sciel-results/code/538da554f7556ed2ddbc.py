import struct
import statistics

# Precomputed constants for 8 rows
# We need P, Q for each row
# Using simple large primes
PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641,
    19349663, 16777619, 15949, 1000003
]
# Q values can be arbitrary offsets
QS = [
    1, 3, 5, 7, 9, 11, 13, 15
]

def _hash_index(item, row):
    p = PRIMES[row]
    q = QS[row]
    h = (item * p + q) & 0xFFFFFFFF
    return h & 63

def _hash_sign(item, row):
    p = PRIMES[row]
    q = QS[row]
    h = (item * p + q) & 0xFFFFFFFF
    # Use bit 6 to determine sign, independent of lower 6 bits
    return 1 if (h & 64) == 0 else -1

def update(state, item):
    # Unpack 8 rows of 64 int16s
    # Total 512 ints, 1024 bytes
    counters = list(struct.unpack('<512h', state))
    
    for r in range(8):
        idx = _hash_index(item, r)
        sign = _hash_sign(item, r)
        offset = r * 64 + idx
        counters[offset] += sign
        
    return struct.pack('<512h', *counters)

def query(state, item):
    counters = list(struct.unpack('<512h', state))
    
    estimates = []
    for r in range(8):
        idx = _hash_index(item, r)
        sign = _hash_sign(item, r)
        offset = r * 64 + idx
        est = sign * counters[offset]
        estimates.append(est)
        
    # Median of 8 values
    med = statistics.median(estimates)
    
    # Ensure non-negative
    if med < 0:
        return 0.0
    return float(med)
