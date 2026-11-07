import struct

# Memory layout: 1024 bytes total
# - Bytes 0..49: 50 bytes for a 400-bit membership bitset
# - Bytes 50..1023: 974 bytes for 974 uint8 counters
#
# Rationale:
# 1. Membership bitset allows returning exact 0 for unseen items, eliminating
#    false positive errors which are significant when querying all 400 items.
# 2. Using uint8 counters allows for ~974 buckets, significantly reducing
#    collision noise compared to 512 uint16 buckets (noise ~1 vs ~2).
# 3. Max expected count in Zipf(1.6) stream is well below 255, so uint8
#    overflow is not a concern for the mean error metric.

BITSET_SIZE = 50
COUNTER_OFFSET = 50
NUM_COUNTERS = 974  # 1024 - 50

def _hash(item):
    # Strong mixing to distribute items 0..399 across 974 buckets
    # Using a large prime multiplier and modulo
    p = 1035493289
    m = 2147483647
    return ((item * p + 31) % m) % NUM_COUNTERS

def _get_bit(state, item):
    byte_idx = item // 8
    bit_idx = item % 8
    return (state[byte_idx] >> bit_idx) & 1

def _set_bit(state, item):
    byte_idx = item // 8
    bit_idx = item % 8
    state[byte_idx] |= (1 << bit_idx)

def update(state, item):
    b = bytearray(state)
    
    # 1. Set membership bit
    _set_bit(b, item)
    
    # 2. Increment counter
    idx = _hash(item)
    offset = COUNTER_OFFSET + idx
    b[offset] = (b[offset] + 1) & 0xFF
    
    return bytes(b)

def query(state, item):
    # 1. Check membership
    if not _get_bit(state, item):
        return 0.0
    
    # 2. Return counter value
    idx = _hash(item)
    offset = COUNTER_OFFSET + idx
    return float(state[offset])
