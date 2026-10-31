import zlib

def priority(vector, n):
    # Convert tuple to bytes for hashing
    # Vector elements are 0, 1, 2
    b = bytes(vector)
    h = zlib.crc32(b)
    return float(h)
