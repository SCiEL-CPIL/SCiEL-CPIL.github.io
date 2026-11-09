import struct

# Layout:
#   Bytes [0, 256): exact uint8 counters for items 0..255 (fixed window)
#   Bytes [256, 1024): 4 rows x 192 slots x uint16 (little-endian) CMS for items 256..399
#   Total: 256 + 4*192*2 = 256 + 1536 ... wait, that's 1792 > 1024.
#   Let me recalculate: 1024 - 256 = 768 bytes for CMS.
#   768 / 2 (uint16) = 384 counters total.
#   With 4 rows: 384 / 4 = 96 slots per row.
#   So: 4 rows x 96 slots x uint16 = 768 bytes. Total = 256 + 768 = 1024. ✓

EXACT_W = 256          # exact counter width for items 0..255
CMS_D = 4              # depth of CMS
CMS_W = 96             # width of each CMS row (96 uint16 counters)
CMS_BYTES = CMS_D * CMS_W * 2  # 768 bytes
PRIME = 2147483647

# Pre-compute hash parameters for CMS (items 256..399, mapped to 0..143 for hashing)
_PS = (1035493289, 775374203, 1495862053, 2015145641)

# Pre-compute CMS indices for items 256..399
# We hash the original item value but modulo CMS_W
_CMS_INDICES = [[0] * CMS_D for _ in range(400)]
for item in range(256, 400):
    for r in range(CMS_D):
        p = _PS[r]
        _CMS_INDICES[item][r] = ((item * p + 31 * r) % PRIME) % CMS_W

def update(state, item):
    """
    Update the state.
    state: bytes of length 1024
    item: int in 0..399
    returns: new state as bytes
    """
    buf = bytearray(state)
    
    if item < EXACT_W:
        # Exact counter: increment byte at position item
        if buf[item] < 255:
            buf[item] += 1
    else:
        # CMS: increment uint16 counters in each row
        idxs = _CMS_INDICES[item]
        for r in range(CMS_D):
            offset = EXACT_W + r * CMS_W * 2 + idxs[r] * 2
            # Read uint16 little-endian
            val = buf[offset] | (buf[offset + 1] << 8)
            if val < 65535:
                val += 1
            buf[offset] = val & 0xFF
            buf[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate (int)
    """
    if item < EXACT_W:
        # Exact count
        return int(state[item])
    else:
        # CMS: return minimum across rows
        idxs = _CMS_INDICES[item]
        min_est = 65536
        for r in range(CMS_D):
            offset = EXACT_W + r * CMS_W * 2 + idxs[r] * 2
            val = state[offset] | (state[offset + 1] << 8)
            if val < min_est:
                min_est = val
        return int(min_est)
