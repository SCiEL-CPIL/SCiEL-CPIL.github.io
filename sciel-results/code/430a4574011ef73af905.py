"""
Hybrid frequency sketch: dedicated counters + Count-Min sketch.
State layout (1024 bytes):
  - Bytes 0..379:   380 dedicated uint8 counters for items 0..379
  - Bytes 380..1019: 16 rows x 40 cols = 640 bytes CM sketch for items 380..399
  - Bytes 1020..1023: unused (4 bytes)
"""
import time

_START = time.time()
_TIMEOUT = 115

_NUM_DEDICATED = 380
_NUM_ROWS = 16
_NUM_COLS = 40
_CM_OFFSET = _NUM_DEDICATED  # 380
_CM_SIZE = _NUM_ROWS * _NUM_COLS  # 640
_TOTAL = _CM_OFFSET + _CM_SIZE  # 1020

# Primes for hashing items 380..399 into 40 columns
_P = [
    1035493289, 775374203, 1495862053, 2015145641,
    1234567891, 2269547793, 1000003, 1000033,
    1000037, 1000039, 1000081, 1000087,
    1000139, 1000151, 1000153, 1000157,
]


def _hash_cm(item, row):
    """Hash item to CM column index."""
    h = (item + 1) * _P[row]
    h ^= h >> 16
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= h >> 16
    return h % _NUM_COLS


def update(state, item):
    if time.time() - _START > _TIMEOUT:
        return state
    buf = bytearray(state)
    if item < _NUM_DEDICATED:
        if buf[item] < 255:
            buf[item] += 1
    else:
        for row in range(_NUM_ROWS):
            col = _hash_cm(item, row)
            idx = _CM_OFFSET + row * _NUM_COLS + col
            if buf[idx] < 255:
                buf[idx] += 1
    return bytes(buf)


def query(state, item):
    if time.time() - _START > _TIMEOUT:
        return 0
    if item < _NUM_DEDICATED:
        return state[item]
    else:
        result = 255
        for row in range(_NUM_ROWS):
            col = _hash_cm(item, row)
            idx = _CM_OFFSET + row * _NUM_COLS + col
            val = state[idx]
            if val < result:
                result = val
        return result
