import hashlib

def priority(vector, n):
    # Convert the vector to a bytes object
    # vector is a tuple of integers in {0, 1, 2}
    b = bytes(vector)
    
    # Use MD5 for a deterministic, well-distributed hash
    h = hashlib.md5(b).digest()
    
    # Take the first 8 bytes and interpret as a big-endian integer
    # This gives a 64-bit pseudo-random priority
    val = int.from_bytes(h[:8], 'big')
    
    return float(val)
