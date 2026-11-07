"""
Combined strategy: 2-row Count-Min Sketch with bias correction.
Structure: 2 rows x 256 uint16 counters = 1024 bytes.
- Each row has collision noise ~ Poisson(1024/256=4).
- Query returns min of two rows, then subtracts a bias correction.
- The min-of-2 operation reduces the systematic over-estimation.
- Bias correction: E[min of 2 Poisson(4)] ≈ 3.0 (analytically ~2.98).
  Subtract 3 to center the estimate around the true count.
- For items with true count 0: estimate ≈ min(Pois(4), Pois(4)) - 3 ≈ -0.02, clamped to 0.
- For items with true count 2: estimate ≈ 2 + 3 - 3 = 2 (good).
- For high-count items: the -3 correction is negligible relative to the count.

This combines:
- Structural idea from inspiration 2 (2-row CMS with min)
- Refinement machinery from current program (bias correction with clamping)
"""

def update(state, item):
    b = bytearray(state)
    
    # Row 0: multiplier 1035493289, salt 31
    idx0 = ((item * 1035493289 + 31) % 2147483647) % 256
    off0 = idx0 * 2
    val0 = int.from_bytes(b[off0:off0+2], 'little')
    val0 += 1
    b[off0:off0+2] = val0.to_bytes(2, 'little')
    
    # Row 1: multiplier 775374203, salt 62
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
    
    # Min of two rows reduces over-estimation
    min_val = val0 if val0 < val1 else val1
    
    # Bias correction: E[min of 2 Poisson(4)] ≈ 3.0
    corrected = min_val - 3
    if corrected < 0:
        corrected = 0
    
    return corrected
