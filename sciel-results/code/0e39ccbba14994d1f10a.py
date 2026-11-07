"""Memory-bounded streaming frequency sketch: exact counters + compact CMS."""
import struct

# Layout of the 1024-byte state:
#   Bytes 0..269  : 270 exact uint8 counters (items 0-19 and 150-399)
#   Bytes 270..1023: CMS sketch for items 20-149 (129 items)
#                   3 rows x 40 columns of uint16 = 240 bytes (offset 270..509)
#                   Remaining 514 bytes are unused (zero)

EXACT_ITEMS = list(range(0, 20)) + list(range(150, 400))  # 270 items
EXACT_OFFSET = 0
EXACT_SIZE = 270

CMS_ROWS = 3
CMS_COLS = 40
CMS_OFFSET = EXACT_SIZE  # 270
CMS_SIZE = CMS_ROWS * CMS_COLS * 2  # 240 bytes

# Hash primes for the CMS (different from seed to avoid correlation)
PRIMES = (2654435761, 40503, 2246822519)
MOD_LARGE = 2147483647

# Precompute exact-item index mapping
_EXACT_IDX = {}
for _i, _item in enumerate(EXACT_ITEMS):
    _EXACT_IDX[_item] = _i


def _is_exact(item):
    """Check if item is in the exact-count set and return its index, or -1."""
    idx = _EXACT_IDX.get(item, -1)
    return idx


def _cms_indices(item):
    """Compute the 3 column indices for an item in the CMS."""
    indices = []
    for row in range(CMS_ROWS):
        p = PRIMES[row]
        idx = ((item * p + 31 * row) % MOD_LARGE) % CMS_COLS
        indices.append(idx)
    return indices


def update(state, item):
    """Update the frequency sketch state with a new item.
    
    Args:
        state: bytes (1024 bytes)
        item: int (0..399)
    Returns:
        new state as bytes (1024 bytes)
    """
    # Work with a bytearray for mutability
    ba = bytearray(state)
    
    idx = _is_exact(item)
    if idx >= 0:
        # Exact counter: increment the uint8 at position EXACT_OFFSET + idx
        pos = EXACT_OFFSET + idx
        ba[pos] = (ba[pos] + 1) & 0xFF
    else:
        # CMS update for items 20-149
        cols = _cms_indices(item)
        for row in range(CMS_ROWS):
            col = cols[row]
            offset = CMS_OFFSET + (row * CMS_COLS + col) * 2
            # Read uint16 little-endian
            val = struct.unpack_from('<H', ba, offset)[0]
            val = (val + 1) & 0xFFFF
            struct.pack_into('<H', ba, offset, val)
    
    return bytes(ba)


def query(state, item):
    """Query the frequency estimate for an item.
    
    Args:
        state: bytes (1024 bytes)
        item: int (0..399)
    Returns:
        nonnegative frequency estimate
    """
    idx = _is_exact(item)
    if idx >= 0:
        # Exact count
        return state[EXACT_OFFSET + idx]
    else:
        # CMS query: minimum over 3 rows
        cols = _cms_indices(item)
        min_val = 0xFFFF
        for row in range(CMS_ROWS):
            col = cols[row]
            offset = CMS_OFFSET + (row * CMS_COLS + col) * 2
            val = struct.unpack_from('<H', state, offset)[0]
            if val < min_val:
                min_val = val
        return min_val
