"""
Hybrid streaming frequency sketch:
- 400-bit membership bitset to identify seen items (return 0 for unseen).
- 256 uint16 counters (1024 - 50 = 974 bytes used for counters, 50 bytes for bitset)
  to store approximate frequencies for seen items.
- For seen items, return the counter value at the hashed bucket.
- For unseen items, return 0.

Memory layout (1024 bytes total):
- Bytes 0..49: 50 bytes for 400-bit bitset (400 bits = 50 bytes)
- Bytes 50..1023: 974 bytes. We use 256 uint16 counters = 512 bytes.
  Actually, let's use the full 1024 bytes efficiently.
  
  Let's reconsider: 400 bits = 50 bytes.
  Remaining: 1024 - 50 = 974 bytes.
  We can fit 487 uint16 counters. But 487 is odd. Let's use 486 uint16 counters (972 bytes) + 2 bytes padding.
  Or simpler: use 256 uint16 counters (512 bytes). That leaves 974 - 512 = 462 bytes unused.
  
  Alternatively, we can use more counters. Let's use 486 uint16 counters.
  But the hash needs to map to 0..485.
  
  Actually, let's stick to a clean design:
  - 50 bytes for bitset (400 bits)
  - 512 bytes for 256 uint16 counters
  - Total: 562 bytes used, 462 bytes unused.
  
  This is fine. The key insight is that the bitset eliminates collision noise for unseen items.
"""

def update(state, item):
    b = bytearray(state)
    
    # Update bitset: set bit for item
    byte_idx = item // 8
    bit_idx = item % 8
    b[byte_idx] |= (1 << bit_idx)
    
    # Update counter: hash item to one of 256 buckets
    # Use a strong hash to spread items 0..399 into 0..255
    # h = (item * 1035493289 + 31) % 2147483647
    # bucket = h % 256
    h = (item * 1035493289 + 31) % 2147483647
    bucket = h % 256
    
    # Counter starts at byte 50
    counter_offset = 50 + bucket * 2
    # Read current uint16 value
    val = b[counter_offset] | (b[counter_offset + 1] << 8)
    val += 1
    # Write back
    b[counter_offset] = val & 0xFF
    b[counter_offset + 1] = (val >> 8) & 0xFF
    
    return bytes(b)


def query(state, item):
    b = bytearray(state)
    
    # Check bitset
    byte_idx = item // 8
    bit_idx = item % 8
    if not (b[byte_idx] & (1 << bit_idx)):
        return 0
    
    # Item is seen, return counter value
    h = (item * 1035493289 + 31) % 2147483647
    bucket = h % 256
    counter_offset = 50 + bucket * 2
    val = b[counter_offset] | (b[counter_offset + 1] << 8)
    return val
