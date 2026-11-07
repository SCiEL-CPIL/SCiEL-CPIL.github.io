"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

def update(state, item):
    # Layout:
    # Bytes 0-63: Exact counts for items 0-63 (uint8)
    # Bytes 64-1023: CMS Sketch (960 bytes)
    # CMS: 8 rows x 30 columns of uint32 (8 * 30 * 4 = 960)
    
    b = bytearray(state)
    
    if item < 64:
        b[item] = (b[item] + 1) & 0xFF
    else:
        # Update CMS for item
        # 8 rows, 30 cols
        # Indices based on simple hash
        # We use 8 different hash functions
        # h_i(item) = (item * p_i + c_i) % 30
        
        # Primes for hashing
        # We need to map item to 0..29
        # item is 64..399
        # Let x = item
        # h0 = (x * 7) % 30
        # h1 = (x * 11) % 30
        # ...
        
        # To be more robust, use larger multipliers
        # and mod 30
        multipliers = [1035493289, 775374203, 1495862053, 2015145641, 
                       1725123659, 1814420481, 1744610631, 1824859077]
        
        for i in range(8):
            idx = (item * multipliers[i]) % 30
            offset = 64 + i * 30 * 4 + idx * 4
            # Read uint32
            val = struct.unpack_from('<I', b, offset)[0]
            val = (val + 1) & 0xFFFFFFFF
            struct.pack_into('<I', b, offset, val)
            
    return bytes(b)

def query(state, item):
    if item < 64:
        return state[item]
    else:
        multipliers = [1035493289, 775374203, 1495862053, 2015145641, 
                       1725123659, 1814420481, 1744610631, 1824859077]
        
        min_val = float('inf')
        for i in range(8):
            idx = (item * multipliers[i]) % 30
            offset = 64 + i * 30 * 4 + idx * 4
            val = struct.unpack_from('<I', state, offset)[0]
            if val < min_val:
                min_val = val
        return min_val
# EVOLVE-BLOCK-END
