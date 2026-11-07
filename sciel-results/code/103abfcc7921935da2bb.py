import struct

# Pre-compute indices for CMS
# Items 64..399
# 7 rows, 128 columns
CMS_ROWS = 7
CMS_COLS = 128
CMS_OFFSET = 128 # 64 * 2 bytes

# Hash function for index generation
def _hash(x, row):
    # Use a simple strong hash
    # splitmix64 variant
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    # Mix in row
    z = (z ^ (row * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z % CMS_COLS

# Pre-compute for items 64..399
# _cms_indices[item] = list of 7 byte offsets (relative to CMS_OFFSET)
_cms_indices = {}
for item in range(64, 400):
    indices = []
    for row in range(CMS_ROWS):
        idx = _hash(item, row)
        indices.append(idx)
    _cms_indices[item] = indices

def update(state, item):
    ba = bytearray(state)
    
    if item < 64:
        # Exact count: uint16 little-endian at offset item * 2
        offset = item * 2
        val = struct.unpack_from('<H', ba, offset)[0]
        val += 1
        struct.pack_into('<H', ba, offset, val)
    else:
        # CMS: uint8 at offset CMS_OFFSET + idx
        indices = _cms_indices[item]
        for idx in indices:
            pos = CMS_OFFSET + idx
            ba[pos] += 1
            if ba[pos] > 255:
                ba[pos] = 255 # Saturate, though counts are small
    
    return bytes(ba)

def query(state, item):
    if item < 64:
        offset = item * 2
        return struct.unpack_from('<H', state, offset)[0]
    else:
        indices = _cms_indices[item]
        min_val = 255
        for idx in indices:
            val = state[CMS_OFFSET + idx]
            if val < min_val:
                min_val = val
        return min_val
