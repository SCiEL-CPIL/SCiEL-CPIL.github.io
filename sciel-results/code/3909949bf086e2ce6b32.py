"""
Structural redesign combining hybrid exact-count + wider sketch:
- 128 bytes: exact uint8 counters for items 0..127 (captures Zipf mass with zero error)
- 896 bytes: 4 rows x 224 uint8 counters for items 128..399 (272 tail items)

Rationale:
- The Zipf(1.6) stream concentrates heavy mass in low item IDs. Giving items 0..127
  exact 1-byte counters eliminates all error for the highest-frequency items.
- The remaining 272 items (128..399) get a 4x224 CM sketch using uint8 counters.
  224 slots for 272 items is tighter than the current 128 slots for 144 items
  (load factor ~1.21 vs ~1.13), but the 4-row depth and uint8 width give good
  collision resistance. The key win is that we freed up bytes by using only 128
  exact counters instead of 256, allowing a wider sketch (224 vs 128) for the tail.
- uint8 saturation at 255 is acceptable: in a 1024-item stream, only the very top
  Zipf items exceed 255, and those are in the exact region (items 0..127).
"""

# Layout
_N_EXACT = 128          # items 0..127 get exact uint8 counters
_SK_OFFSET = _N_EXACT   # 128
_SK_DEPTH = 4
_SK_WIDTH = 224         # 4 * 224 * 1 = 896 bytes
_SK_BYTES = _SK_DEPTH * _SK_WIDTH
_TOTAL = _N_EXACT + _SK_BYTES  # 1024

# Hash parameters for the 4-row CM sketch
_PRIME = 2147483647
_PS = (1035493289, 775374203, 1495862053, 2015145641)
_B = (31, 67, 129, 193)

# Precompute sketch indices for tail items (128..399)
_SK_IDX = [[0] * _SK_DEPTH for _ in range(400)]
for item in range(_N_EXACT, 400):
    for r in range(_SK_DEPTH):
        _SK_IDX[item][r] = ((item * _PS[r] + _B[r]) % _PRIME) % _SK_WIDTH


def update(state, item):
    buf = bytearray(state)

    if item < _N_EXACT:
        # Exact counter: 1 byte at offset item
        cur = buf[item]
        if cur < 255:
            buf[item] = cur + 1
    else:
        # Sketch: increment 4 counters (one per row)
        idxs = _SK_IDX[item]
        for r in range(_SK_DEPTH):
            offset = _SK_OFFSET + r * _SK_WIDTH + idxs[r]
            cur = buf[offset]
            if cur < 255:
                buf[offset] = cur + 1

    return bytes(buf)


def query(state, item):
    if item < _N_EXACT:
        # Exact count
        return state[item]
    else:
        # Minimum over 4 rows
        idxs = _SK_IDX[item]
        min_val = 255
        for r in range(_SK_DEPTH):
            offset = _SK_OFFSET + r * _SK_WIDTH + idxs[r]
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val
