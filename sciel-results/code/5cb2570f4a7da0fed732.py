import zlib

def priority(vector, n):
    # Create a stable hash of the vector
    # vector is a tuple of ints
    h = 0
    for v in vector:
        h = (h * 31 + v) & 0xFFFFFFFF
    # Use crc32 for better distribution
    data = bytes(vector)
    crc = zlib.crc32(data)
    # Map to float
    return float(crc) / 0xFFFFFFFF
