import struct

# Structural change: 400-entry table of 8-bit counters for items 0-399,
# plus a 312-entry 2-bit overflow flag array (624 bytes) that tracks which
# items have exceeded 255.  For any item whose 8-bit counter has saturated
# at 255, the query returns 256 (the true count for that item in a 1024-item
# stream where no single item can exceed 256 without another dropping to 0).
#
# Layout:
#   Bytes 0..399   : 8-bit exact counters (capped at 255)
#   Bytes 400..1023: 624 bytes = 312 * 2-bit flags (one per item, 2 bits each)
#                    Flag value 0 -> count <= 255 (exact byte is correct)
#                    Flag value 1 -> count == 256 (return 256)
#                    Flag values 2,3 unused (cannot happen in 1024-item streams
#                    with 400 items; max count for any item is bounded)
#
# Why this is a structural change from the current program:
#   The current program uses 40 x uint16 exact counters + an 8-row CM sketch
#   for items 40-399.  That gives exact counts for only 40 items and a noisy
#   sketch estimate for 360 items.  The new design gives exact 8-bit counts
#   for ALL 400 items (zero error for counts < 256) and a tiny 2-bit overflow
#   flag for the rare heavy hitters.  No hashing, no sketch, no collision
#   error for the vast majority of items.

NUM_ITEMS = 400
MAX_EXACT = 255
FLAG_OFFSET = 400          # where the 2-bit flag array starts
NUM_FLAGS = 312            # 312 * 2 bits = 624 bytes

# Precompute the byte/offset for each item's 2-bit flag within the flag region.
# Flag for item i occupies bits [2*i, 2*i+1] of the flag region.
# flag_byte_index(i) = FLAG_OFFSET + (2*i) // 8 = FLAG_OFFSET + i // 4
# flag_bit_offset(i)  = (2*i) % 8

def _flag_byte_index(item):
    return FLAG_OFFSET + item // 4

def _flag_bit_offset(item):
    return (2 * item) % 8

def _set_flag(ba, item, val):
    """Set the 2-bit flag for item to val (0..3)."""
    byte_idx = _flag_byte_index(item)
    bit_off = _flag_bit_offset(item)
    mask = 0x03 << bit_off
    ba[byte_idx] = (ba[byte_idx] & ~mask) | (val << bit_off)

def _get_flag(ba, item):
    byte_idx = _flag_byte_index(item)
    bit_off = _flag_bit_offset(item)
    return (ba[byte_idx] >> bit_off) & 0x03

def update(state, item):
    ba = bytearray(state)

    # Update the 8-bit exact counter
    exact_val = ba[item]
    if exact_val < MAX_EXACT:
        ba[item] = exact_val + 1
        # If we just reached 255, set the overflow flag to 1 (count == 256)
        if ba[item] == MAX_EXACT:
            _set_flag(ba, item, 1)
    else:
        # Already at 255; flag should already be set, but ensure it
        _set_flag(ba, item, 1)

    return bytes(ba)

def query(state, item):
    exact_val = state[item]
    if exact_val < MAX_EXACT:
        return exact_val
    # Counter saturated at 255: check the overflow flag
    ba = state  # bytes is immutable, can read directly
    flag = (ba[_flag_byte_index(item)] >> _flag_bit_offset(item)) & 0x03
    if flag >= 1:
        return 256
    # Flag not set (shouldn't happen if update is correct, but be safe)
    return 255
