"""
Hybrid structural change: precomputed-index Count-Min Sketch with 4 rows x 128 uint16.

Structural idea from inspiration 2: precompute all 400*4 hash indices and byte
offsets at module load so update/query are pure table lookups (no modular
arithmetic at runtime). This removes per-call overhead and is safe because the
item domain is fixed (0..399).

Refinement from current program / inspiration 1: keep the wider-row preference
that empirically helped (W=256, D=2) but push toward a balanced 4x128 layout.
Rationale: with 400 distinct items, W=128 gives load ~3.1 per counter, while
D=4 gives a tighter min-of-4 upper bound. The seed used 4x64 (load 6.25);
doubling width to 128 should cut collision bias substantially, and the extra
row over D=2 should tighten the estimate for items that do collide.

Layout: 4 rows x 128 counters x 2 bytes = 1024 bytes exactly.

Hash parameters: four distinct large odd multipliers with distinct offsets,
chosen to spread items 0..399 well across 128 slots.
"""

import struct

_D = 4
_W = 128
_N_ITEMS = 400

# Verify: _D * _W * 2 = 4 * 128 * 2 = 1024 bytes
assert _D * _W * 2 == 1024

_M = 2147483647  # 2^31 - 1

# (a, b) hash parameters for each row
_PARAMS = [
    (1035493289, 31),
    (775374203, 67),
    (1495862053, 97),
    (2015145641, 127),
]

# Precompute byte offsets: _OFF[row * N_ITEMS + item] = byte offset of counter
_OFFSETS = [0] * (_D * _N_ITEMS)

def _build_offsets():
    for row in range(_D):
        a, b = _PARAMS[row]
        row_base = row * (_W * 2)
        for item in range(_N_ITEMS):
            idx = (a * item + b) % _M % _W
            _OFFSETS[row * _N_ITEMS + item] = row_base + idx * 2

_build_offsets()


def update(state, item):
    """Update the sketch state by incrementing counters for the given item."""
    buf = bytearray(state)
    item_base = item  # item is 0..399, already in range
    for row in range(_D):
        off = _OFFSETS[row * _N_ITEMS + item_base]
        val = struct.unpack_from('<H', buf, off)[0]
        if val < 65535:
            val += 1
        struct.pack_into('<H', buf, off, val)
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    min_val = 65536
    for row in range(_D):
        off = _OFFSETS[row * _N_ITEMS + item]
        val = struct.unpack_from('<H', state, off)[0]
        if val < min_val:
            min_val = val
    return min_val
