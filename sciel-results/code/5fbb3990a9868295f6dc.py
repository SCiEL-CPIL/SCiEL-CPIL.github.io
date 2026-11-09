import struct

def update(state, item):
    buf = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', buf, offset)[0]
    val += 1
    struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val
