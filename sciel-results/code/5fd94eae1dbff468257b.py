"""
Hybrid frequency sketch: exact counters for known heavy hitters + multi-row CMS for the rest.
State layout (1024 bytes):
  - First 400 bytes: 100 uint32 exact counters for items in [0..19] and [150..399] (20+250=270... wait)
  Actually: items 0-19 (20 items) + items 150-399 (250 items) = 270 items.
  But 270 * 4 = 1080 > 1024. So we can't do exact for all.
  
  Let me reconsider. The key insight from the knowledge:
  - Stream 3: 512 uniform on 0-19, 512 uniform on 150-399
  - Stream 2: Zipf(1.6) mod 400 - heavy hitters in low range
  - Stream 1: uniform on 0-399
  
  The measured improvement came from "exact counter allocation to cover high-frequency items across all three streams (0-19 and 150-399)"
  
  But 270 items * 4 bytes = 1080 bytes > 1024. So we need to be smarter.
  
  Alternative: Use a smaller exact set. Items 0-19 (20 items) are hit in streams 2 and 3.
  Items 150-399 (250 items) are hit in stream 3.
  
  What if we use uint16 for some? Max count per item: 1024 items total, so max count is 1024 which fits in uint16.
  
  Let's allocate:
  - 20 uint32 counters for items 0-19 (80 bytes)
  - 250 uint16 counters for items 150-399 (500 bytes)
  Total exact: 580 bytes
  Remaining: 1024 - 580 = 444 bytes for CMS on items 20-149 (130 items)
  
  444 / 4 = 111 uint32 counters for CMS.
  With 130 items in that range, we want good collision resistance.
  111 counters: could do 3 rows x 37 cols, or 4 rows x 27 cols, etc.
  
  Actually, let me think about this differently. The scoring is mean |estimate - true| / 1024 over all 400 items.
  
  For items that are exactly counted, error = 0.
  For items in the CMS range (20-149), we get approximation error.
  
  Stream 1 (uniform 0-399): each item appears ~1024/400 = 2.56 times on average.
  Stream 2 (Zipf 1.6 mod 400): heavy hitters in low range, items 20-149 get fewer hits.
  Stream 3: items 20-149 get 0 hits.
  
  So items 20-149 only appear in streams 1 and 2. In stream 1, ~2.56 each. In stream 2, Zipf means lower items get more.
  
  For the CMS part, items 20-149 (130 items) with total insertions from streams 1+2 into that range:
  Stream 1: 1024 * 130/400 = 332.8
  Stream 2: depends on Zipf, but roughly maybe 100-200 in that range.
  Total ~430-530 insertions into 130 items, so average ~3-4 per item.
  
  With 111 counters and good hashing, error should be small.
  
  Let me design:
  - Bytes 0-79: 20 uint32 counters for items 0-19
  - Bytes 80-579: 250 uint16 counters for items 150-399
  - Bytes 580-1023: 444 bytes = 111 uint32 counters for CMS on items 20-149
  
  For CMS with 111 counters, let's use 3 rows x 37 columns.
  Hash: for item in 20-149, map to 0-129, then hash into 37 columns.
  
  Actually, let me use a better hash. For each row r, index = (item * prime_r + offset_r) % 37.
  
  Primes: use large odd numbers.
  
  Query:
  - If item in 0-19: return exact uint32 count
  - If item in 150-399: return exact uint16 count
  - If item in 20-149: return min of 3 CMS row values
"""

import struct

# Layout constants
EXACT_LOW_START = 0      # bytes 0-79: 20 uint32 for items 0-19
EXACT_HIGH_START = 80    # bytes 80-579: 250 uint16 for items 150-399
CMS_START = 580          # bytes 580-1023: 111 uint32 for CMS

CMS_ROWS = 3
CMS_COLS = 37
CMS_TOTAL = CMS_ROWS * CMS_COLS  # 111

# Primes for CMS hashing
CMS_PRIMES = [1035493289, 775374203, 1495862053]
CMS_OFFSETS = [31, 63, 95]


def update(state, item):
    """Update the sketch state with one more occurrence of item."""
    new_state = bytearray(state)
    
    if 0 <= item <= 19:
        # Exact uint32 counter
        offset = EXACT_LOW_START + item * 4
        val = struct.unpack_from('<I', new_state, offset)[0]
        struct.pack_into('<I', new_state, offset, val + 1)
    elif 150 <= item <= 399:
        # Exact uint16 counter
        idx = item - 150
        offset = EXACT_HIGH_START + idx * 2
        val = struct.unpack_from('<H', new_state, offset)[0]
        struct.pack_into('<H', new_state, offset, val + 1)
    elif 20 <= item <= 149:
        # CMS update
        for r in range(CMS_ROWS):
            col = (item * CMS_PRIMES[r] + CMS_OFFSETS[r]) % CMS_COLS
            offset = CMS_START + (r * CMS_COLS + col) * 4
            val = struct.unpack_from('<I', new_state, offset)[0]
            struct.pack_into('<I', new_state, offset, val + 1)
    # else: item out of range, do nothing
    
    return bytes(new_state)


def query(state, item):
    """Query the estimated frequency of item."""
    if 0 <= item <= 19:
        offset = EXACT_LOW_START + item * 4
        return struct.unpack_from('<I', state, offset)[0]
    elif 150 <= item <= 399:
        idx = item - 150
        offset = EXACT_HIGH_START + idx * 2
        return struct.unpack_from('<H', state, offset)[0]
    elif 20 <= item <= 149:
        # CMS query: min across rows
        min_val = float('inf')
        for r in range(CMS_ROWS):
            col = (item * CMS_PRIMES[r] + CMS_OFFSETS[r]) % CMS_COLS
            offset = CMS_START + (r * CMS_COLS + col) * 4
            val = struct.unpack_from('<I', state, offset)[0]
            if val < min_val:
                min_val = val
        return min_val if min_val != float('inf') else 0
    else:
        return 0
