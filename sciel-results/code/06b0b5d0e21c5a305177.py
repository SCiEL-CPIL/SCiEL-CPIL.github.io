import struct

_STATE_SIZE = 1024
_NUM_ITEMS = 400

def _ensure_state(state):
    if state is None:
        return b'\x00' * _STATE_SIZE
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    if len(state) < _STATE_SIZE:
        state = state + b'\x00' * (_STATE_SIZE - len(state))
    elif len(state) > _STATE_SIZE:
        state = state[:_STATE_SIZE]
    return state

def update(state, item):
    item = int(item)
    if item < 0 or item >= _NUM_ITEMS:
        return _ensure_state(state) # Return unchanged if invalid item? Or clamp?
        # Task says items are 0..399.
    
    state = _ensure_state(state)
    buf = bytearray(state)
    off = item * 2
    # Use struct for safety and speed?
    # struct.unpack_from('<H', buf, off)
    # But manual is fine.
    val = buf[off] | (buf[off + 1] << 8)
    if val < 65535:
        val += 1
        buf[off] = val & 0xFF
        buf[off + 1] = (val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    item = int(item)
    if item < 0 or item >= _NUM_ITEMS:
        return 0
        
    state = _ensure_state(state)
    off = item * 2
    return state[off] | (state[off + 1] << 8)
