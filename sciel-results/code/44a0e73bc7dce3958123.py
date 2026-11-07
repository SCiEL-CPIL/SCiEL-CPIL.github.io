import struct

def update(state, item):
    offset = item * 2
    new_state = bytearray(state)
    current = struct.unpack_from('<H', state, offset)[0]
    struct.pack_into('<H', new_state, offset, current + 1)
    return bytes(new_state)

def query(state, item):
    offset = item * 2
    return struct.unpack_from('<H', state, offset)[0]
