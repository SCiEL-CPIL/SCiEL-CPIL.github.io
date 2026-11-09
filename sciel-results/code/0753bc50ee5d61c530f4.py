import numpy as np
import time

# Two-tier CMS with heavy-hitter correction
# Tier 1: items 0-99  -> 256 counters (int16) = 512 bytes, depth 4
# Tier 2: items 100-399 -> 128 counters (int16) = 256 bytes, depth 4
# Heavy hitters: 256 bytes reserved for correction (up to 32 items * 8 bytes each)
# Total: 512 + 256 + 256 = 1024 bytes

_T1_W = 256   # width for tier 1 (items 0-99)
_T2_W = 128   # width for tier 2 (items 100-399)
_D = 4        # depth for both tiers

# Hash params for tier 1
_A1 = np.array([1035493289, 775374203, 1495862053, 2015145641], dtype=np.uint64)
_B1 = np.array([31, 63, 97, 131], dtype=np.uint64)
_P = np.uint64(2147483647)

# Hash params for tier 2 (different primes to avoid correlation)
_A2 = np.array([1000003, 1000033, 1000037, 1000039], dtype=np.uint64)
_B2 = np.array([7, 11, 13, 17], dtype=np.uint64)

# Layout:
# Bytes 0..511:   tier1 counters (256 x int16)
# Bytes 512..767: tier2 counters (128 x int16)
# Bytes 768..1023: heavy hitter table (32 entries x 8 bytes: 4-byte item, 4-byte count)

_HH_SLOTS = 32
_HH_OFFSET = 768

def _t1_indices(item):
    x = np.uint64(item)
    hashed = (_A1 * x + _B1) % _P
    return hashed % _T1_W

def _t2_indices(item):
    x = np.uint64(item - 100)
    hashed = (_A2 * x + _B2) % _P
    return hashed % _T2_W

def _get_hh_counts(state):
    """Extract heavy hitter counts from state bytes."""
    hh_bytes = state[_HH_OFFSET:]
    counts = {}
    for i in range(_HH_SLOTS):
        item = int.from_bytes(hh_bytes[i*8:i*8+4], 'little', signed=False)
        cnt = int.from_bytes(hh_bytes[i*8+4:i*8+8], 'little', signed=False)
        if item < 400 and cnt > 0:
            counts[item] = cnt
    return counts

def _update_hh_table(state, item, true_count):
    """Update heavy hitter table. Returns new state bytes."""
    hh_bytes = bytearray(state[_HH_OFFSET:])
    # Find slot for this item
    found = False
    for i in range(_HH_SLOTS):
        it = int.from_bytes(hh_bytes[i*8:i*8+4], 'little', signed=False)
        if it == item:
            new_cnt = min(true_count, 0xFFFFFFFF)
            hh_bytes[i*8:i*8+4] = item.to_bytes(4, 'little')
            hh_bytes[i*8+4:i*8+8] = new_cnt.to_bytes(4, 'little')
            found = True
            break
    if not found:
        # Find empty slot or replace smallest
        best_i = 0
        best_cnt = 0xFFFFFFFF
        for i in range(_HH_SLOTS):
            it = int.from_bytes(hh_bytes[i*8:i*8+4], 'little', signed=False)
            cnt = int.from_bytes(hh_bytes[i*8+4:i*8+8], 'little', signed=False)
            if it >= 400 or cnt == 0:
                best_i = i
                best_cnt = 0
                break
            if cnt < best_cnt:
                best_cnt = cnt
                best_i = i
        hh_bytes[best_i*8:best_i*8+4] = item.to_bytes(4, 'little')
        hh_bytes[best_i*8+4:best_i*8+8] = min(true_count, 0xFFFFFFFF).to_bytes(4, 'little')
    
    return state[:_HH_OFFSET] + bytes(hh_bytes)

def update(state, item):
    buf = bytearray(state)
    
    # Update tier 1 or tier 2 counters
    if item < 100:
        arr = np.frombuffer(buf, dtype=np.int16, count=_T1_W, offset=0)
        indices = _t1_indices(item)
        for idx in indices:
            arr[idx] += 1
        buf[0:_T1_W*2] = arr.tobytes()
    else:
        arr = np.frombuffer(buf, dtype=np.int16, count=_T2_W, offset=_T1_W*2)
        indices = _t2_indices(item)
        for idx in indices:
            arr[idx] += 1
        buf[_T1_W*2:_T1_W*2+_T2_W*2] = arr.tobytes()
    
    # Update heavy hitter table: estimate current count to decide if worth tracking
    # Use the CMS estimate as a proxy
    if item < 100:
        t1_arr = np.frombuffer(buf, dtype=np.int16, count=_T1_W, offset=0)
        est = int(np.min(t1_arr[_t1_indices(item)]))
    else:
        t2_arr = np.frombuffer(buf, dtype=np.int16, count=_T2_W, offset=_T1_W*2)
        est = int(np.min(t2_arr[_t2_indices(item)]))
    
    # Only track items with count >= 3 in heavy hitter table
    if est >= 3:
        # Get current stored count for this item if any
        hh_counts = _get_hh_counts(bytes(buf))
        current = hh_counts.get(item, est - 1)  # est includes this increment
        new_count = est  # Use the CMS estimate as the "true" count approximation
        buf = bytearray(_update_hh_table(bytes(buf), item, new_count))
    
    return bytes(buf)

def query(state, item):
    # Check heavy hitter table first
    hh_counts = _get_hh_counts(state)
    if item in hh_counts:
        return float(hh_counts[item])
    
    # Otherwise use CMS estimate
    if item < 100:
        arr = np.frombuffer(state, dtype=np.int16, count=_T1_W, offset=0)
        indices = _t1_indices(item)
        vals = arr[indices]
    else:
        arr = np.frombuffer(state, dtype=np.int16, count=_T2_W, offset=_T1_W*2)
        indices = _t2_indices(item)
        vals = arr[indices]
    
    return float(max(0, int(np.min(vals))))
