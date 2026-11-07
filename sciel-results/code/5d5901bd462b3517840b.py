"""Exact direct-addressing frequency counters for 400 items in 1024 bytes."""
import struct

# Layout: bytes 0..799 -> 400 uint16 counters (2 bytes each, little-endian)
# bytes 800..1023 -> reserved (kept zero for compatibility / future use)

def update(state, item):
    """Increment the exact counter for `item` (0..399). Returns new state bytes."""
    # Work on a mutable copy to avoid aliasing issues
    buf = bytearray(state)
    idx = item * 2  # each counter is 2 bytes
    # Read current 16-bit little-endian value
    cur = buf[idx] | (buf[idx + 1] << 8)
    # Increment, cap at 65535 (won't be reached in this task: max count ~1024)
    new = cur + 1
    if new > 0xFFFF:
        new = 0xFFFF
    buf[idx] = new & 0xFF
    buf[idx + 1] = (new >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    """Return the exact frequency estimate for `item` (0..399)."""
    idx = item * 2
    val = state[idx] | (state[idx + 1] << 8)
    return float(val)
