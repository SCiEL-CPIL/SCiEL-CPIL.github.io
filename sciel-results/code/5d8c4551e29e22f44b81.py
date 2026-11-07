"""
Structural change: Hybrid exact + CM sketch.

State layout (1024 bytes):
- Bytes 0-255: 256 dedicated uint8 counters for items 0-255 (exact counting).
- Bytes 256-1023: 768 bytes for a Count-Min sketch (12 rows x 16 columns of uint8).

Rationale:
- Stream 1 (uniform 0..399): All items have low counts (~2-3). Items 0-255 are exact.
  Items 256-399 use CM. With 12 rows x 16 cols, expected collision error per cell is
  total/16 = 1024/16 = 64, but since true counts are ~2-3, the min of 12 rows will
  typically be close to the true count (collisions add ~64 expected, but min of 12
  reduces this). Actually wait - for uniform stream, items 256-399 have true count ~2-3.
  The CM sketch cells will have values around 2-3 + collisions. With 16 columns and
  1024 total items, expected collision per cell is 1024/16 = 64. That's way too high.
  
  Hmm, this is a problem. Let me reconsider.

  For the CM sketch part, we're only tracking items 256-399 (144 items). But the sketch
  doesn't know which items go where - it hashes all items. Wait, no - we only update
  the sketch for items 256-399. So the total count in the sketch is at most 1024 (if
  all items are in 256-399) but typically much less.

  For uniform stream: ~144*1024/400 ≈ 368 items go to 256-399. Expected collision per
  cell: 368/16 ≈ 23. True count per item: ~1024/400 ≈ 2.56. So estimate ≈ 2.56 + 23 = 25.6.
  That's bad.

  For Zipf: Most mass is in items 0-255 (exact). Items 256-399 have very low counts.
  For uniform on 150-399 (stream 3 second half): items 256-399 get 512/150 ≈ 3.4 each.
  Items 150-255 are exact. So sketch gets 512/150 * (144/150) ≈ 3.4 * 0.96 ≈ 3.3 items per item.
  Wait, 512 items uniform on 150..399 = 250 items. So 512/250 ≈ 2.05 per item.
  Items 256-399: 144 items, so 512 * 144/250 ≈ 295 items in sketch. Expected collision: 295/16 ≈ 18.4.
  True count per item: ~2.05. Estimate ≈ 2.05 + 18.4 ≈ 20.5. Still bad.

  The issue is that 16 columns is too few. Let me reconsider the layout.

  Alternative: Use more columns for the sketch.
  - Bytes 0-199: 200 uint8 counters for items 0-199 (exact).
  - Bytes 200-1023: 824 bytes for sketch.
  
  824 / 2 = 412 uint16 counters. 412 = 4 * 103 or similar.
  Or 824 = 8 * 103 (8 rows x 103 cols uint8).
  
  With 103 columns: expected collision for uniform stream items 200-399 (200 items):
  1024 * 200/400 / 103 ≈ 512/103 ≈ 5. True count: 2.56. Estimate ≈ 7.56. Better.
  
  For stream 3: items 200-399 in second half (200 items out of 250): 512 * 200/250 = 410.
  Collision: 410/103 ≈ 4. True count: 2.05. Estimate ≈ 6.

  Still not great. The fundamental issue is that with limited bytes, we can't have both
  exact counters AND a good sketch.

  Let me try a different structural approach: use the full 1024 bytes as a CM sketch
  with wider counters, but optimize the hash functions specifically for this task.

  Actually, the key insight from the inspiration is: "exact counting for items 0-255 
  (covers all Zipf heavy hitters and stream 3's first half) eliminates overestimation 
  for high-frequency items, while 12x16 CM sketch for items 256-399 provides better 
  variance than 8x64 for the tail"

  But my analysis shows 12x16 has high collision. Let me re-examine.

  Wait - maybe the trick is that for the CM sketch, we only update it for items 256-399.
  So the total mass in the sketch is much less than 1024.

  For uniform stream: items 256-399 = 144 items out of 400. Expected items in sketch:
  1024 * 144/400 ≈ 368.64. With 16 columns, expected collision per cell: 368.64/16 ≈ 23.
  True count per item: 1024/400 ≈ 2.56. So estimate ≈ 2.56 + 23 ≈ 25.6. Error ≈ 23.
  Score contribution: 23/1024 ≈ 0.0225 per item. For 144 items: 144 * 0.0225 / 400 ≈ 0.0081.

  For Zipf: items 256-399 have very low probability. Sum of probabilities for items 256-399:
  Very small, maybe 0.01-0.02. So expected items in sketch: 1024 * 0.015 ≈ 15.
  Collision per cell: 15/16 ≈ 1. True count per item: ~0.015 * 1024/144 ≈ 0.1.
  Estimate ≈ 0.1 + 1 ≈ 1.1. Error ≈ 1. For 144 items: 144 * 1/1024 / 400 ≈ 0.00035.

  For stream 3: first 512 uniform on 0-19 (all exact), second 512 uniform on 150-399.
  Items 256-399 in second half: 144 items out of 250. Expected: 512 * 144/250 ≈ 295.
  Collision per cell: 295/16 ≈ 18.4. True count: 512/250 ≈ 2.05. Estimate ≈ 20.4.
  Error ≈ 18.4. For 144 items: 144 * 18.4/1024/400 ≈ 0.0064.

  Total average error across streams: roughly (0.0081 + 0.00035 + 0.0064)/3 ≈ 0.00495.
  That's close to the observed 0.004112!

  So the 12x16 sketch for items 256-399 does work because the total mass in the sketch
  is much less than 1024 for most streams.

  Let me implement this:
  - Bytes 0-255: uint8 counters for items 0-255
  - Bytes 256-1023: 12 rows x 16 columns of uint8 for CM sketch (items 256-399 only)
  """

import struct

_NUM_EXACT = 256
_SKETCH_ROWS = 12
_SKETCH_COLS = 16

# Primes for hashing (need 12 distinct primes)
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    1784567893,
    2147483629,
    1567891237,
    987654321,
    314159265,
    271828182,
    161803398,
]
_MOD = 2147483647


def _sketch_hash(item, row):
    """Hash item to a column in the sketch. Only called for items 256-399."""
    p = _PRIMES[row]
    return ((item * p + 31 * row) % _MOD) % _SKETCH_COLS


def update(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: new state as bytes
    """
    buf = bytearray(state)
    
    if item < _NUM_EXACT:
        # Exact counter for items 0-255
        if buf[item] < 255:
            buf[item] += 1
    else:
        # CM sketch for items 256-399
        base_offset = _NUM_EXACT
        for row in range(_SKETCH_ROWS):
            col = _sketch_hash(item, row)
            offset = base_offset + row * _SKETCH_COLS + col
            if buf[offset] < 255:
                buf[offset] += 1
    
    return bytes(buf)


def query(state, item):
    """
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate
    """
    if item < _NUM_EXACT:
        # Exact count
        return state[item]
    else:
        # CM sketch: take min of all rows
        base_offset = _NUM_EXACT
        best = 255
        for row in range(_SKETCH_ROWS):
            col = _sketch_hash(item, row)
            offset = base_offset + row * _SKETCH_COLS + col
            val = state[offset]
            if val < best:
                best = val
        return best
