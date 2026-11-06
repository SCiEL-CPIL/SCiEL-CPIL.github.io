import zlib

def priority(vector, n):
    # Convert tuple to bytes
    # Assuming vector elements are 0, 1, 2
    b = bytes(vector)
    return zlib.crc32(b) / 4294967296.0
