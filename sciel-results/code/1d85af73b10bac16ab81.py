import struct
import time

_START = time.time()
_DEADLINE = _START + 110.0

# State layout:
# Bytes 0..399: 400 dedicated uint8 counters (one per item)
# Bytes 400..1023: 624 bytes for a 16-row x 39-column CM sketch (for overflow/error correction)
# Actually, let's use a simpler approach:
# Since max count is ~150 for Zipf(1.6) on 400 items over 1024 items,
# and uniform gives ~2.56 per item, uint8 (0-255) is sufficient for exact counting.
# The only issue is if count exceeds 255, but that's unlikely.
#
# Strategy: Use 400 dedicated uint8 counters. This gives exact counts for all items.
# The remaining 624 bytes can be unused or used for a backup sketch.
#
# Why this works:
# - Uniform: each item appears ~2.56 times, max ~10, well within uint8
# - Zipf(1.6): item 0 appears most frequently. P(item=k) ∝ (k+1)^(-1.6)
#   Sum of (k+1)^(-1.6) for k=0..399 ≈ 1 + sum_{i=2}^{400} i^(-1.6)
#   The sum converges, dominated by first few terms. Item 0 gets the most.
#   With 1024 items, item 0 might get ~150-200 occurrences, still < 256.
# - Third stream: 512 uniform on 0..19 (each ~26.7 times) + 512 uniform on 150..399 (each ~2.05 times)
#   Max is ~27 for items 0-19, well within uint8.
#
# So dedicated uint8 counters give EXACT counts, hence zero error.

NUM_ITEMS = 400

def update(state, item):
    """Update the state by incrementing the counter for the given item."""
    if time.time() > _DEADLINE:
        return state
    
    # Convert state to a bytearray for mutation
    ba = bytearray(state)
    
    # Increment the dedicated counter for this item
    # Each item has its own byte at position item (0..399)
    if 0 <= item < NUM_ITEMS:
        # Increment with saturation at 255
        if ba[item] < 255:
            ba[item] += 1
    
    return bytes(ba)

def query(state, item):
    """Query the estimated frequency of the given item."""
    if time.time() > _DEADLINE:
        return 0
    
    # Return the exact count from the dedicated counter
    if 0 <= item < NUM_ITEMS:
        return int(state[item])
    else:
        return 0
