"""
Hybrid frequency estimation:
- 400 exact counters (1 byte each) for all possible items (0..399)
- Remaining 624 bytes used for a Count-Min Sketch as backup

Since the item space is only 400 and we have 1024 bytes, we can store
exact counts for all items using 1 byte each (max count per item is 1024,
which fits in a uint16, but let's check: 3 streams of 1024 items = 3072 total,
but each item appears at most... in uniform over 400 items with 1024 draws,
expected count is ~2.56, max is small. In Zipf(1.6), the top item could get
many hits. Let's use uint16 for exact counters to be safe.

Wait: 400 items * 2 bytes = 800 bytes for exact counters.
Remaining: 1024 - 800 = 224 bytes for a CMS.

Actually, let me reconsider. With 400 items and 1024 bytes:
- Option A: 400 * 2 = 800 bytes for exact uint16 counters, 224 bytes for CMS
- Option B: 400 * 1 = 400 bytes for exact uint8 counters (if max count < 256), 624 bytes for CMS

For Zipf(1.6) on 400 items with 1024 draws: the probability of item 0 is
1/(0+1)^1.6 / sum(1/(i+1)^1.6 for i in 0..399). The sum is approximately
zeta(1.6) ≈ 2.85. So P(item 0) ≈ 1/2.85 ≈ 0.35. Expected count ≈ 359.
That exceeds 255! So uint8 won't work for the top Zipf items.

Let's use uint16 for exact counters: 400 * 2 = 800 bytes.
Remaining 224 bytes: 112 uint16 counters. We can do a CMS with e.g. 4 rows x 28 cols,
or 7 rows x 16 cols, etc.

Actually, since we have exact counts for ALL 400 items, the CMS is redundant.
The query just returns the exact count. This should give near-perfect accuracy.

But wait - the score is |estimate - true_count| / 1024, mean over 400 items, mean over 3 streams.
If we have exact counts, the error is 0 for all items in each stream. Score = 0.

This should be a massive improvement. Let me implement it.
"""
# EVOLVE-BLOCK-START
import struct

def update(state, item):
    """
    State layout:
    - Bytes 0..799: 400 uint16 LE exact counters for items 0..399
    - Bytes 800..1023: 224 bytes reserved (unused, kept for compatibility)
    """
    # Read and increment the exact counter for this item
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    new_val = val + 1
    # Pack back
    new_state = bytearray(state)
    struct.pack_into('<H', new_state, offset, new_val)
    return bytes(new_state)

def query(state, item):
    """
    Return the exact count for the item.
    """
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return float(val)
# EVOLVE-BLOCK-END
