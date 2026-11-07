import struct

# Precomputed constants for 4 rows of 128 uint16 counters
# Total size: 4 * 128 * 2 = 1024 bytes
PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641
]
# Offsets to ensure different hash sequences
QS = [
    1, 3, 5, 7
]

def update(state, item):
    # Unpack 512 uint16s
    counters = list(struct.unpack('<512H', state))
    
    for r in range(4):
        p = PRIMES[r]
        q = QS[r]
        h = (item * p + q) & 0xFFFFFFFF
        idx = h & 127
        offset = r * 128 + idx
        counters[offset] += 1
        
    return struct.pack('<512H', *counters)

def query(state, item):
    # Unpack 512 uint16s
    counters = struct.unpack('<512H', state)
    
    min_est = float('inf')
    for r in range(4):
        p = PRIMES[r]
        q = QS[r]
        h = (item * p + q) & 0xFFFFFFFF
        idx = h & 127
        offset = r * 128 + idx
        val = counters[offset]
        if val < min_est:
            min_est = val
            
    return float(min_est)
