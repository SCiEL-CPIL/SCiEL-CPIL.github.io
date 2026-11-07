import struct

# 4 rows of 128 uint16 counters
# 4 * 128 * 2 = 1024 bytes

# Hash parameters
# We need 4 hash functions h_i(x) = (a_i * x + b_i) % 128
# Using large primes to ensure good distribution
A = [1035493289, 775374203, 1495862053, 2015145641]
B = [31, 63, 95, 127] # Just some constants

def update(state, item):
    # Unpack 512 uint16s
    data = struct.unpack('<512H', state)
    data = list(data)
    
    for i in range(4):
        idx = (A[i] * item + B[i]) % 128
        row_offset = i * 128
        data[row_offset + idx] += 1
        
    return struct.pack('<512H', *data)

def query(state, item):
    data = struct.unpack('<512H', state)
    
    min_val = 65535 # Max uint16
    for i in range(4):
        idx = (A[i] * item + B[i]) % 128
        row_offset = i * 128
        val = data[row_offset + idx]
        if val < min_val:
            min_val = val
            
    return float(min_val)
