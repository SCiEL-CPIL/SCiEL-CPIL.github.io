"""Hybrid frequency estimator: exact counters for hot items + Count-Min Sketch for the rest.

Layout (1024 bytes total):
- Bytes 0..79:   20 x uint32 LE exact counters for items 0..19
- Bytes 80..1023: 8 rows x 30 cols x uint32 LE CMS counters for items 20..399
  (8 * 30 * 4 = 960 bytes; total = 80 + 960 = 1040... wait that's over)

Let me recalculate:
- 20 hot items x 4 bytes = 80 bytes
- Remaining: 1024 - 80 = 944 bytes
- 944 / 4 = 236 counters
- 7 rows x 33 cols = 231 counters = 924 bytes -> total 1004
- 8 rows x 29 cols = 232 counters = 928 bytes -> total 1008

Let's use 7 rows x 33 cols = 924 bytes, total 1004. Safe.

Actually, let's reconsider. The key insight from inspiration 1 is that for a uniform
stream of 1024 items over 400 values, max count is ~25, and for Zipf(1.6) the hot
items 0..19 will have very high counts. Using exact counters for the 20 hottest items
eliminates all collision noise for those items, which dominate the error in Zipf and
Mixed streams.

For the remaining 380 items, we use a CMS with enough rows/cols to keep collision
noise low. With 7 rows x 33 cols, each counter is uint32.

Hash: H_j(i) = ((i * P_j + B_j) % 2147483647) % 33
"""

import struct

HOT_ITEMS = 20
CM_ROWS = 7
CM_COLS = 33
CM_OFFSET = HOT_ITEMS * 4  # 80

P_CONSTS = (
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1000000007,
    1000000009,
    1000000021,
)

B_CONSTS = (
    31,
    57,
    91,
    133,
    199,
    257,
    311,
)

MOD_LARGE = 2147483647


def _cm_index(item, row):
    return ((item * P_CONSTS[row] + B_CONSTS[row]) % MOD_LARGE) % CM_COLS


def update(state, item):
    buf = bytearray(state)
    
    if item < HOT_ITEMS:
        offset = item * 4
        val = struct.unpack_from('<I', buf, offset)[0]
        val += 1
        struct.pack_into('<I', buf, offset, val)
    else:
        for row in range(CM_ROWS):
            idx = _cm_index(item, row)
            offset = CM_OFFSET + (row * CM_COLS + idx) * 4
            val = struct.unpack_from('<I', buf, offset)[0]
            val += 1
            struct.pack_into('<I', buf, offset, val)
    
    return bytes(buf)


def query(state, item):
    if item < HOT_ITEMS:
        offset = item * 4
        val = struct.unpack_from('<I', state, offset)[0]
        return float(val)
    else:
        min_val = float('inf')
        for row in range(CM_ROWS):
            idx = _cm_index(item, row)
            offset = CM_OFFSET + (row * CM_COLS + idx) * 4
            val = struct.unpack_from('<I', state, offset)[0]
            if val < min_val:
                min_val = val
        return float(min_val)
