import hashlib

def priority(vector, n):
    # Deterministic pseudo-random priority via SHA-256
    # vector is a tuple of ints in {0,1,2}, length n
    data = bytes(vector)
    h = hashlib.sha256(data).digest()
    # Use first 8 bytes as a 64-bit integer, normalize to [0, 1)
    val = int.from_bytes(h[:8], 'big')
    return val / (2**64)
