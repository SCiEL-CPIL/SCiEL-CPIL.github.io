"""New approach: dense Count-Min sketch, 8 rows x 32 uint32 counters,
splitmix64-based high-quality hashing, min-of-rows query.
Uses the full 1024-byte budget; different structure + hash family vs seed."""

import struct

# 8 rows, 32 columns each -> 256 uint32 counters -> 1024 bytes
ROWS = 8
COLS = 32
M = 1 << 64
MASK64 = (1 << 64) - 1
MOD = 2147483647  # large prime, reused as intermediate modulus for variety

def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    x = x ^ (x >> 31)
    return x

# Per-row salt constants (distinct 64-bit seeds)
_SALT = [
    0x8A2F6B1C3D4E5F60,
    0x1F3C5A7B9D2E4F68,
    0x7C9E1B3D5F7A9C2E,
    0x3E5F7A9C1B3D5E7F,
    0x5A7C9E1B3D5F7A9C,
    0x9C1E3B5D7F9A1C3E,
    0xB1D3F5A7C9E1B3D5,
    0xD5F7A9C1E3B5D7F9,
]

def _index(item, row):
    # Combine item and row salt, then splitmix for a well-mixed 64-bit value,
    # then map to [0, COLS). Using modulo a prime first reduces low-bit bias.
    base = (item * 2654435761 + _SALT[row]) & MASK64
    h = _splitmix64(base)
    return h % COLS

def update(state, item):
    # state is bytes of length 1024 (256 uint32 LE counters)
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
