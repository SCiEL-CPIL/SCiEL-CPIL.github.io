"""Exact per-item byte counters for all 400 items."""
import struct

# EVOLVE-BLOCK-START
# State layout: 400 bytes, one byte per item (0..399), storing exact count.
# Remaining 624 bytes are unused (zeros).
# Max possible count: 1024 (if all items were the same), but in practice
# for these distributions, max count is well under 255.
# Even in worst case, if count exceeds 255, we wrap around (mod 256).
# But let's check: uniform on 0..399 with 1024 items -> expected count ~2.56, max maybe ~10-15.
# Zipf(1.6) mod 400: item 0 gets the most. Zipf freq for rank 1 is 1/H(400,1.6).
# H(400,1.6) ~ sum_{i=1}^{400} 1/i^1.6. Approx integral from 1 to 400 of x^-1.6 dx = [x^-0.6 / -0.6]_1^400 = (1 - 400^-0.6)/0.6.
# 400^0.6 = (4*100)^0.6 = 4^0.6 * 100^0.6 ≈ 2.297 * 15.85 ≈ 36.4. So 400^-0.6 ≈ 0.0275.
# H ≈ (1 - 0.0275)/0.6 ≈ 1.62. So item 0 gets ~1024/1.62 ≈ 632. That's > 255!
# So byte counters will overflow for Zipf stream item 0.
# 
# Wait, the task says "Zipf(1.6) modulo 400". Let me re-read.
# "Zipf(1.6) modulo 400" means we generate Zipf-distributed items in 1..400 (or 0..399?) and take mod 400.
# Typically Zipf is 1-indexed. So items 1..400, then mod 400 gives 0..399.
# Item 1 (becomes 1 mod 400 = 1) has highest frequency.
# Actually, if we use scipy.stats.zipf, it's 1-indexed. So zipf.rvs(1.6, size=1024) gives values >= 1.
# Mod 400: values 1..399 map to themselves, 400 maps to 0, 401 maps to 1, etc.
# So item 1 and item 401 (->1) and 801 (->1) all contribute to item 1.
# The dominant term is still the first few items.
# Let's calculate more carefully.
# P(item=k) for Zipf(1.6) ~ k^-1.6 / H(1.6) where H is the harmonic number of order 1.6.
# For large N, H(1.6) ≈ 1/0.6 + 0.5*1 + ... ≈ 1.667 + 0.5 = 2.167? No, the Euler-Maclaurin formula.
# Actually, let's just note that item 1 (and multiples of 400) will have the highest count.
# If max count > 255, byte counters wrap.
# 
# Alternative: Use uint16 counters? 400 * 2 = 800 bytes. Fits in 1024.
# Max count 1024 fits in uint16 (max 65535).
# This gives exact counts for all items.
#
# Let's use uint16 LE counters for all 400 items.
# State layout: 400 * 2 = 800 bytes for counters. Remaining 224 bytes unused.

def update(state, item):
    state = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    val += 1
    struct.pack_into('<H', state, offset, val)
    return bytes(state)

def query(state, item):
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return float(val)
# EVOLVE-BLOCK-END
