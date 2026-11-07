import struct

def update(state, item):
    # state is 1024 bytes
    # 4 rows, 128 cols, uint16
    # Layout: Row 0 (256 bytes), Row 1 (256 bytes), ...
    
    # Unpack 512 uint16s
    counters = list(struct.unpack('<512H', state))
    
    p = (1035493289, 775374203, 1495862053, 2015145641)
    
    for r in range(4):
        idx = ((item * p[r] + 31 * r) % 2147483647) % 128
        offset = r * 128 + idx
        counters[offset] += 1
        
    return struct.pack('<512H', *counters)

def query(state, item):
    counters = list(struct.unpack('<512H', state))
    p = (1035493289, 775374203, 1495862053, 2015145641)
    
    min_val = float('inf')
    for r in range(4):
        idx = ((item * p[r] + 31 * r) % 2147483647) % 128
        offset = r * 128 + idx
        val = counters[offset]
        if val < min_val:
            min_val = val
            
    return min_val
