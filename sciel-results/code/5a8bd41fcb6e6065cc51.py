def update(state, item):
    item = int(item)
    # Ensure state is bytes
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    buf = bytearray(state)
    # Pad if necessary? No, assume 1024.
    off = item * 2
    val = buf[off] | (buf[off + 1] << 8)
    if val < 65535:
        val += 1
        buf[off] = val & 0xFF
        buf[off + 1] = (val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    item = int(item)
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    off = item * 2
    return state[off] | (state[off + 1] << 8)
