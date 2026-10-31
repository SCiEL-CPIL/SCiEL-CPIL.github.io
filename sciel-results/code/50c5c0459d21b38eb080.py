import zlib

def priority(vector, n):
    # Convert vector to bytes for CRC32
    # Vector elements are 0, 1, 2
    b = bytes(vector)
    return float(zlib.crc32(b))
