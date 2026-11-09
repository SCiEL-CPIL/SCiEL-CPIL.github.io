"""
High-precision hybrid sketch:
- Items 0..159 stored exactly as uint16 in a direct-address table (320 bytes).
- Items 160..399 handled by a 3-row x 384-col Count-Min sketch with uint16 (2304? no, 3*384*2=2304 > 1024).
  
Wait, budget is 1024 bytes total.
Let's recalculate:
- DAT for 0..159: 160 items * 2 bytes = 320 bytes.
- Remaining: 1024 - 320 = 704 bytes.
- CM sketch with uint16: 704 / 2 = 352 counters.
- We can do Depth=2, Width=176 (2*176=352). Or Depth=1, Width=352.
- Better: Depth=3, Width=117 (3*117=351, 1 byte left). Or Depth=2, Width=176.
- Let's use Depth=2, Width=176 for items 160..399 (240 items).
  240 items into 176 bins per row -> moderate collision.
  
Alternative:
- DAT for 0..19 (20 items * 2 = 40 bytes) — covers the first half of stream 3.
- DAT for 150..199 (50 items * 2 = 100 bytes) — covers the transition zone of stream 3.
- That's 140 items in DAT = 280 bytes.
- Remaining 744 bytes for sketch: 372 counters.
- Depth=2, Width=186 (2*186=372).
- Items 20..149 and 200..399 go to sketch.
  
Actually, the simplest and most robust approach given the streams:
Stream 1: Uniform 0..399
Stream 2: Zipf(1.6) mod 400 — heavily skewed toward low items
Stream 3: 512 uniform 0..19, 512 uniform 150..399

The Zipf stream means items 0..~50 appear very frequently. Stream 3 also hits 0..19 heavily.
So items 0..60 are "hot". Let's give them exact storage.

Plan:
- DAT for items 0..63 (64 items * 2 bytes = 128 bytes).
- Remaining: 896 bytes = 448 counters.
- CM sketch: Depth=2, Width=224 (2*224=448).
- Items 64..399 (336 items) mapped into 224 bins per row.

This is similar to the current program but let's optimize the hash and ensure correctness.
The current program already does this. Why was it rejected? "empty" reason, likely a code issue.

Let me write a clean, correct implementation:
- 1024 bytes total.
- Bytes 0..127: DAT for items 0..63 (uint16 LE).
- Bytes 128..1023: CM sketch, 2 rows × 224 cols, uint16 LE.

update: if item < 64, increment DAT; else increment 2 CM cells.
query: if item < 64, return DAT value; else return min of 2 CM cells.
"""

_DAT_SIZE = 64
_DAT_BYTES = _DAT_SIZE * 2  # 128
_CM_DEPTH = 2
_CM_WIDTH = 224
_CM_BYTES = _CM_DEPTH * _CM_WIDTH * 2  # 896
_TOTAL = _DAT_BYTES + _CM_BYTES  # 1024

# Hash parameters
_A = (1035493289, 775374203)
_B = (0, 129)

def _read16(buf, off):
    return buf[off] | (buf[off + 1] << 8)

def _write16(buf, off, val):
    buf[off] = val & 0xFF
    buf[off + 1] = (val >> 8) & 0xFF

def update(state, item):
    buf = bytearray(state)
    if item < _DAT_SIZE:
        off = item * 2
        v = _read16(buf, off)
        _write16(buf, off, v + 1)
    else:
        for i in range(_CM_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            off = _DAT_BYTES + i * (_CM_WIDTH * 2) + idx * 2
            v = _read16(buf, off)
            _write16(buf, off, v + 1)
    return bytes(buf)

def query(state, item):
    if item < _DAT_SIZE:
        return _read16(state, item * 2)
    else:
        mn = 65535
        for i in range(_CM_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            off = _DAT_BYTES + i * (_CM_WIDTH * 2) + idx * 2
            v = _read16(state, off)
            if v < mn:
                mn = v
        return mn
