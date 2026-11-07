import struct

def update(state, item):
    b = bytearray(state)
    offset = item * 2
    # Unpack uint16
    val = struct.unpack_from('<H', b, offset)[0]
    val += 1
    struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    offset = item * 2
    return struct.unpack_from('<H', state, offset)[0]
