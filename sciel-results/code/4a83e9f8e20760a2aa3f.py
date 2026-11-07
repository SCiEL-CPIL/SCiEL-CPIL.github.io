"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

def update(state, item):
    # 4 rows of 64 uint32 counters = 1024 bytes
    # Same hash as seed: ((item * p + 31 * row) % 2147483647) % 64
    b = bytearray(state)
    p_values = (1035493289, 775374203, 1495862053, 2015145641)
    
    for row in range(4):
        p = p_values[row]
        idx = ((item * p + 31 * row) % 2147483647) % 64
        offset = row * 256 + idx * 4
        # Read uint32 little-endian
        val = struct.unpack_from('<I', b, offset)[0]
        val = (val + 1) & 0xFFFFFFFF
        struct.pack_into('<I', b, offset, val)
    
    return bytes(b)

def query(state, item):
    p_values = (1035493289, 775374203, 1495862053, 2015145641)
    min_val = 0xFFFFFFFF
    
    for row in range(4):
        p = p_values[row]
        idx = ((item * p + 31 * row) % 2147483647) % 64
        offset = row * 256 + idx * 4
        val = struct.unpack_from('<I', state, offset)[0]
        if val < min_val:
            min_val = val
            
    return min_val
# EVOLVE-BLOCK-END
