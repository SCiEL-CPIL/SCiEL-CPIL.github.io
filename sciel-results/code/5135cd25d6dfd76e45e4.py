import struct

def update(state, item):
    ba = bytearray(state)
    offset = item * 2
    val = ba[offset] | (ba[offset + 1] << 8)
    val += 1
    ba[offset] = val & 0xFF
    ba[offset + 1] = (val >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    offset = item * 2
    val = state[offset] | (state[offset + 1] << 8)
    return val
