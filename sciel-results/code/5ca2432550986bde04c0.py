"""
Optimized streaming frequency sketch: 2-row Count-Min Sketch.
Strategy: 2 rows x 256 uint16 counters = 1024 bytes.
Query returns min of the two row estimates (classic CMS bias reduction).
Rationale:
- Per-row collision noise: N/W = 1024/256 = 4.
- Min-of-2 independent rows reduces systematic over-estimation.
- Two independent strong hash multipliers ensure row independence.
- uint16 is sufficient (max count <= 1024).
"""

def update(state, item):
    b = bytearray(state)
    
    # Row 0: multiplier 1035493289
    idx0 = ((item * 1035493289 + 31) % 2147483647) % 256
    off0 = idx0 * 2
    val0 = int.from_bytes(b[off0:off0+2], 'little')
    val0 += 1
    b[off0:off0+2] = val0.to_bytes(2, 'little')
    
    # Row 1: multiplier 775374203
    idx1 = ((item * 775374203 + 62) % 2147483647) % 256
    off1 = 512 + idx1 * 2
    val1 = int.from_bytes(b[off1:off1+2], 'little')
    val1 += 1
    b[off1:off1+2] = val1.to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    # Row 0
    idx0 = ((item * 1035493289 + 31) % 2147483647) % 256
    off0 = idx0 * 2
    val0 = int.from_bytes(state[off0:off0+2], 'little')
    
    # Row 1
    idx1 = ((item * 775374203 + 62) % 2147483647) % 256
    off1 = 512 + idx1 * 2
    val1 = int.from_bytes(state[off1:off1+2], 'little')
    
    return min(val0, val1)
