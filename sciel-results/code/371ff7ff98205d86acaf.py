"""Memory-bounded streaming frequency sketch: dual-mode exact + CM sketch."""
import struct
import time

# Constants
NUM_ITEMS = 400
STATE_SIZE = 1024

# Layout:
# Bytes 0..399:   400 exact 8-bit counters (capped at 255)
# Bytes 400..511: 128 bytes = 32 x uint32 for CM sketch row 0 (64 counters... wait)
# Let me recalculate.
# 400 bytes for exact counters
# Remaining: 1024 - 400 = 624 bytes
# Use 4 rows of 64 uint16 counters: 4 * 64 * 2 = 512 bytes
# Total: 400 + 512 = 912 bytes. We have 112 bytes left.
# Let's use 4 rows of 64 uint16 = 512 bytes for CM sketch.
# Plus maybe some high-bit extensions for heavy hitters.

# Actually, let me reconsider the layout for maximum effectiveness:
# The key insight: with 400 items and 1024 items per stream, the expected count per item
# in uniform stream is 1024/400 = 2.56. In Zipf, top items can have counts ~50+.
# In the third stream, items 0-19 get 512/20 = 25.6 expected, items 150-399 get 512/250 = 2.05.

# Strategy: 
# - 400 bytes: exact 8-bit counters (max 255). For items with count > 255, we lose precision.
# - But wait, can any item exceed 255? In Zipf(1.6) mod 400, item 0 has highest probability.
#   P(item=0) in Zipf(1.6) over 1..400: the Zipf distribution over items 1..400, then mod 400.
#   Actually "Zipf(1.6) modulo 400" likely means: sample from Zipf(1.6) over some range, take mod 400.
#   Or it could mean items are drawn from a Zipf distribution on {0,...,399} with parameter 1.6.
#   Let's assume it's Zipf on {0,...,399} where P(k) ∝ 1/(k+1)^1.6 for k=0..399.
#   Then P(0) = 1 / sum_{j=0}^{399} 1/(j+1)^1.6 = 1 / H(400, 1.6) where H is the generalized harmonic number.
#   H(400, 1.6) ≈ sum_{j=1}^{400} j^{-1.6} ≈ (400^{0.4})/0.4 + zeta(1.6) ≈ 400^0.4/0.4 + 2.087
#   400^0.4 = (4*100)^0.4 = 4^0.4 * 100^0.4 ≈ 1.741 * 6.31 ≈ 10.98
#   So H ≈ 10.98/0.4 + 2.087 ≈ 27.45 + 2.087 ≈ 29.54
#   P(0) ≈ 1/29.54 ≈ 0.0339
#   Expected count for item 0 in 1024 items: 1024 * 0.0339 ≈ 34.7
#   So max count is well under 255. 8-bit counters should be sufficient for exact counting!

# Wait, but the problem says "Zipf(1.6) modulo 400". This might mean: generate Zipf(1.6) samples
# (which are positive integers) and take mod 400. In that case, item 0 gets contributions from
# all multiples of 400 in the Zipf distribution, which could be significant.
# But even so, the total probability mass landing on any single residue class mod 400 is bounded.

# Given the uncertainty, let's be safe: use 8-bit exact counters (max 255) for all 400 items.
# If a count exceeds 255, we'll cap it. The error from capping at 255 when true count is, say, 300
# would be 45, which divided by 1024 is ~0.044. That's not great but acceptable as a fallback.

# For the CM sketch portion, let's use the remaining 624 bytes.
# 4 rows x 64 columns x 2 bytes (uint16) = 512 bytes. That leaves 112 bytes unused.
# We could use 4 rows x 78 columns x 2 bytes = 624 bytes. But 78 is not a power of 2, making
# indexing harder. Let's stick with 64 columns and use the extra space wisely.

# Actually, let me reconsider: what if I use a different allocation?
# What about: 400 exact counters (8-bit) + 4 rows of 80 uint16 counters?
# 4 * 80 * 2 = 640 bytes. Total = 400 + 640 = 1040 > 1024. Too much.
# 4 * 78 * 2 = 624. Total = 1024. Perfect! But 78 columns makes hashing awkward.
# Let me just use 64 columns and leave 112 bytes as zeros (or use them for something else).

# Alternative: use the 112 extra bytes for high-bit extensions of the most likely heavy hitters.
# Items 0-19 in the third stream, and top Zipf items, are likely heavy hitters.
# But this complicates the design. Let's keep it simple first.

# Final layout:
# Bytes 0..399:   400 x uint8 exact counters
# Bytes 400..911: 4 rows x 64 cols x uint16 CM sketch counters (512 bytes)
# Bytes 912..1023: 112 bytes reserved (zero)

CM_START = 400
CM_ROWS = 4
CM_COLS = 64
CM_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
CM_MOD_LARGE = 2147483647

def _cm_index(item, row):
    """Compute the column index for item in the given row."""
    p = CM_PRIMES[row]
    return ((item * p + 31 * row) % CM_MOD_LARGE) % CM_COLS

def update(state, item):
    """Update the sketch state with a new item."""
    # Convert to bytearray for mutation
    ba = bytearray(state)
    
    # Update exact counter (8-bit, capped at 255)
    if ba[item] < 255:
        ba[item] += 1
    
    # Update CM sketch counters
    for row in range(CM_ROWS):
        col = _cm_index(item, row)
        offset = CM_START + row * CM_COLS * 2 + col * 2
        val = struct.unpack_from('<H', ba, offset)[0]
        if val < 65535:
            val += 1
        struct.pack_into('<H', ba, offset, val)
    
    return bytes(ba)

def query(state, item):
    """Query the estimated frequency of an item."""
    # Get exact count
    exact_count = state[item]
    
    # Get CM sketch estimates
    cm_estimates = []
    for row in range(CM_ROWS):
        col = _cm_index(item, row)
        offset = CM_START + row * CM_COLS * 2 + col * 2
        val = struct.unpack_from('<H', state, offset)[0]
        cm_estimates.append(val)
    
    cm_min = min(cm_estimates)
    
    # Combine: use the max of exact and CM min, since exact is unbiased (no overcounting)
    # but CM min can overestimate due to collisions.
    # Actually, exact count is always <= true count (if capped) and CM min is >= true count.
    # The best estimate is min(cm_min, some_upper_bound) but we don't have a good upper bound.
    # Let's just return the max of exact and a scaled CM estimate.
    # Actually, since exact is exact (unless capped), and CM min is an overestimate,
    # the true count is between exact_count and cm_min.
    # Return the midpoint? Or just exact_count if it's not capped?
    
    # If exact_count < 255, it's likely exact (or very close). Return it.
    # If exact_count == 255, it might be capped. Use CM min as a better estimate.
    if exact_count < 255:
        return float(exact_count)
    else:
        # Capped: use CM min as the estimate
        return float(cm_min)
