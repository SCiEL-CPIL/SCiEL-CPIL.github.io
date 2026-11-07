    def update(state, item):
        buf = bytearray(state)
        off = item * 2
        val = buf[off] | (buf[off + 1] << 8)
        if val < 65535:
            val += 1
            buf[off] = val & 0xFF
            buf[off + 1] = (val >> 8) & 0xFF
        return bytes(buf)

    def query(state, item):
        off = item * 2
        return state[off] | (state[off + 1] << 8)
    