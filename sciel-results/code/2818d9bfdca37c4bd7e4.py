"""Memory-bounded streaming frequency sketch: hybrid bitset + 4-row Count-Min."""
import struct
import time

# Deadline guard
_T0 = time.monotonic()
_DEADLINE = _T0 + 145.0  # 145s wall-clock limit (5s margin before 150s kill)

# Layout: 512 bytes bitset (400 bits) + 512 bytes for 4 rows x 32 uint32 counters
# Actually: 400 bits = 50 bytes. Let's use:
#   bytes 0..49:   bitset for items 0..399 (50 bytes = 400 bits)
#   bytes 50..511: 4 rows of 64 uint32 counters? That's 4*64*4 = 1024 bytes. Too much.
# Let me recalculate.
#
# 1024 bytes total.
# Bitset: 400 bits = 50 bytes.
# Remaining: 974 bytes.
# 4 rows of N uint32 counters: 4*N*4 <= 974 => N <= 60.875 => N=60.
# 4*60*4 = 960 bytes. Total: 50+960 = 1010 bytes. OK, under 1024.
#
# Actually let's use 4 rows of 64 counters but with uint32: 4*64*4=1024. Plus 50 for bitset = 1074. Over.
#
# Alternative: 4 rows of 48 uint32: 4*48*4 = 768. Plus 50 = 818. Fine.
# Or: 4 rows of 60 uint32: 4*60*4 = 960. Plus 50 = 1010. Fine.
#
# Let's use 4 rows of 64 counters but with a different packing.
# 64 counters * 4 bytes = 256 bytes per row. 4 rows = 1024. Plus 50 for bitset = 1074. Over.
#
# Let's use 4 rows of 48 uint32 counters: 4*48*4=768 bytes. Bitset: 50 bytes. Total: 818 bytes.
# Or better: use the full budget.
# 1024 - 50 = 974 bytes for counters. 974/4 = 243.5 uint32s. 243/4 = 60.75 rows of 4.
# So 4 rows of 60 uint32 counters = 960 bytes. Total 1010.
#
# Actually, let me reconsider. The seed uses 4 rows of 64 uint32 = 1024 bytes exactly.
# If I add a bitset, I need to reduce.
#
# Plan:
#   Bytes 0..49: bitset (400 bits, 50 bytes)
#   Bytes 50..1009: 4 rows of 60 uint32 counters (960 bytes)
#   Bytes 1010..1023: unused (14 bytes)
#
# Total: 1024 bytes.

BITSET_BYTES = 50  # 400 bits
NUM_ROWS = 4
NUM_COUNTERS = 60  # per row
COUNTER_BYTES = NUM_ROWS * NUM_COUNTERS * 4  # 960
TOTAL_BYTES = BITSET_BYTES + COUNTER_BYTES  # 1010, pad to 1024

# Hash parameters (same as seed)
_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647


def _idx(item, row):
    return ((item * _P[row] + 31 * row) % _MOD) % NUM_COUNTERS


def _bit_offset(item):
    """Return (byte_offset, bit_mask) for item in the bitset."""
    byte_off = item >> 3
    bit_mask = 1 << (item & 7)
    return byte_off, bit_mask


def update(state, item):
    """Update the sketch state with a new item occurrence."""
    if time.monotonic() > _DEADLINE:
        return state  # safety: return unchanged

    # Work on a copy
    new_state = bytearray(state)

    # Set the bit in the bitset
    byte_off, bit_mask = _bit_offset(item)
    new_state[byte_off] |= bit_mask

    # Increment counters in each row
    for row in range(NUM_ROWS):
        idx = _idx(item, row)
        # Counter location: BITSET_BYTES + row * NUM_COUNTERS * 4 + idx * 4
        pos = BITSET_BYTES + row * NUM_COUNTERS * 4 + idx * 4
        val = struct.unpack_from('<I', new_state, pos)[0]
        val += 1
        struct.pack_into('<I', new_state, pos, val)

    return bytes(new_state)


def query(state, item):
    """Return a nonnegative frequency estimate for the given item."""
    if time.monotonic() > _DEADLINE:
        return 0  # safety

    # Check bitset: if item was never seen, return 0
    byte_off, bit_mask = _bit_offset(item)
    if not (state[byte_off] & bit_mask):
        return 0

    # Otherwise, return the minimum of the 4 row estimates
    min_est = 2**32 - 1
    for row in range(NUM_ROWS):
        idx = _idx(item, row)
        pos = BITSET_BYTES + row * NUM_COUNTERS * 4 + idx * 4
        val = struct.unpack_from('<I', state, pos)[0]
        if val < min_est:
            min_est = val

    return min_est
