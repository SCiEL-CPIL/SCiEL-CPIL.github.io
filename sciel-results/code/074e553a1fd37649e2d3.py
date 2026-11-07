"""New approach: 2-row Count-Min sketch with 128 columns each.
Uses the full 1024-byte budget (256 uint32 counters).
High-quality splitmix64 hashing.
Min-of-rows query.
This structure (2x128) offers a better bias-variance trade-off for low-frequency
items compared to 4x64 or 8x32, as the wider columns reduce collision noise
significantly while the second row helps reduce variance."""

import struct

# 2 rows, 128 columns each -> 256 uint32 counters -> 1024 bytes
ROWS = 2
COLS = 128
MASK64 = (1 << 64) - 1

def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    x = x ^ (x >> 31)
    return x

# Per-row salt constants
_SALT = [
    0x8A2F6B1C3D4E5F60,
    0x1F3C5A7B9D2E4F68,
]

def _index(item, row):
    base = (item * 2654435761 + _SALT[row]) & MASK64
    h = _splitmix64(base)
    return h % COLS

def update(state, item):
    counters = struct.unpack('<256I', state)
    counters = list(counters)
    for row in range(ROWS):
        idx = row * COLS + _index(item, row)
        counters[idx] = (counters[idx] + 1) & 0xFFFFFFFF
    return struct.pack('<256I', *counters)

def query(state, item):
    counters = struct.unpack('<256I', state)
    best = None
    for row in range(ROWS):
        idx = row * COLS + _index(item, row)
        v = counters[idx]
        if best is None or v < best:
            best = v
    return float(best)
