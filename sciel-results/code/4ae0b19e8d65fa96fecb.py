import struct

# 2 rows, 256 cols, uint16
# 2 * 256 * 2 = 1024 bytes

def update(state, item):
    b = bytearray(state)
    # Row 0
    idx0 = (item * 1035493289) & 255
    off0 = idx0 * 2
    val0 = struct.unpack_from('<H', b, off0)[0]
    struct.pack_into('<H', b, off0, val0 + 1)
    
    # Row 1
    idx1 = (item * 775374203) & 255
    off1 = 512 + idx1 * 2  # Second row starts at byte 512
    val1 = struct.unpack_from('<H', b, off1)[0]
    struct.pack_into('<H', b, off1, val1 + 1)
    
    return bytes(b)

def query(state, item):
    b = bytearray(state)
    idx0 = (item * 1035493289) & 255
    off0 = idx0 * 2
    val0 = struct.unpack_from('<H', b, off0)[0]
    
    idx1 = (item * 775374203) & 255
    off1 = 512 + idx1 * 2
    val1 = struct.unpack_from('<H', b, off1)[0]
    
    return float(min(val0, val1))
