import zlib

def priority(vector, n):
    b = bytes(vector)
    return zlib.crc32(b) / 2**32
