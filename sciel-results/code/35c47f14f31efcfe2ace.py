import struct

# Primes for hashing
PRIMES = [
    1035493289, 
    775374203, 
    1495862053, 
    2015145641,
    1234567891, 
    876543211, 
    543219877, 
    321987653
]
MOD = 2147483647
W = 64
D = 8

def update(state, item):
    # state is bytes, length 1024
    # Convert to mutable list or bytearray
    buf = bytearray(state)
    
    for r in range(D):
        # Calculate index
        # h = (item * P[r] + r * 31) % MOD % W
        # Using 31 as row offset factor as in seed, or just r
        idx = ((item * PRIMES[r] + r * 31) % MOD) % W
        offset = r * W * 2 + idx * 2
        
        # Read current value
        val = struct.unpack_from('<H', buf, offset)[0]
        # Increment
        val += 1
        # Write back
        struct.pack_into('<H', buf, offset, val)
        
    return bytes(buf)

def query(state, item):
    buf = state
    min_val = float('inf')
    
    for r in range(D):
        idx = ((item * PRIMES[r] + r * 31) % MOD) % W
        offset = r * W * 2 + idx * 2
        val = struct.unpack_from('<H', buf, offset)[0]
        if val < min_val:
            min_val = val
            
    return min_val
