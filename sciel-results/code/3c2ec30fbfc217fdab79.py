import hashlib
import struct

def priority(vector, n):
    b = bytes(vector)
    h = hashlib.sha256(b).digest()
    # Take first 8 bytes as unsigned 64-bit int
    val = struct.unpack('<Q', h[:8])[0]
    return float(val)
