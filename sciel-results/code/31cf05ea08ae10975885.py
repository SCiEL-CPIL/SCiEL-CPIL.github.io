import struct
import time

_START = time.monotonic()
_DEADLINE = _START + 145.0  # safety margin before 150s kill

# --- Constants ---
NUM_ROWS = 4
NUM_COLS = 100  # 4 * 100 * 2 = 800 bytes for CM sketch (uint16)
HH_SIZE = 50    # 2 + 50*4 = 202 bytes for heavy hitter table
STATE_SIZE = 1024

# Prime multipliers for hashing (same as seed)
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
M = 2147483647  # 2^31 - 1

def _hash_index(item, row):
    """Compute CM sketch column index using the given hash."""
    return ((item * PRIMES[row] + 31 * row) % M) % NUM_COLS

def _parse_state(state):
    """Parse the 1024-byte state into components."""
    if len(state) != STATE_SIZE:
        raise ValueError(f"State must be {STATE_SIZE} bytes, got {len(state)}")
    
    # Heavy hitter count (uint16 LE at offset 0)
    hh_count = struct.unpack_from('<H', state, 0)[0]
    hh_count = min(hh_count, HH_SIZE)
    
    # Heavy hitter entries: each 4 bytes (item: uint16 LE, count: uint16 LE)
    hh_entries = []
    for i in range(hh_count):
        offset = 2 + i * 4
        item_id, count = struct.unpack_from('<HH', state, offset)
        hh_entries.append((item_id, count))
    
    # CM sketch: starts at offset 2 + HH_SIZE * 4 = 202
    cm_offset = 2 + HH_SIZE * 4
    cm_data = state[cm_offset:cm_offset + NUM_ROWS * NUM_COLS * 2]
    
    return hh_count, hh_entries, cm_data

def _build_state(hh_count, hh_entries, cm_data):
    """Build the 1024-byte state from components."""
    state = bytearray(STATE_SIZE)
    struct.pack_into('<H', state, 0, hh_count)
    
    for i, (item_id, count) in enumerate(hh_entries):
        offset = 2 + i * 4
        struct.pack_into('<HH', state, offset, item_id, count)
    
    cm_offset = 2 + HH_SIZE * 4
    state[cm_offset:cm_offset + NUM_ROWS * NUM_COLS * 2] = cm_data
    
    return bytes(state)

def update(state, item):
    """Update the sketch state with a new item."""
    if len(state) != STATE_SIZE:
        # If state is invalid, return a fresh zero state
        return b'\x00' * STATE_SIZE
    
    hh_count, hh_entries, cm_data = _parse_state(state)
    
    # Check deadline
    if time.monotonic() > _DEADLINE:
        return state
    
    # Convert CM data to a list of lists for easier manipulation
    cm = []
    for row in range(NUM_ROWS):
        row_data = []
        for col in range(NUM_COLS):
            offset = row * NUM_COLS * 2 + col * 2
            val = struct.unpack_from('<H', cm_data, offset)[0]
            row_data.append(val)
        cm.append(row_data)
    
    # Update CM sketch
    for row in range(NUM_ROWS):
        col = _hash_index(item, row)
        cm[row][col] = min(cm[row][col] + 1, 65535)
    
    # Update heavy hitter table
    found = False
    new_entries = []
    min_idx = -1
    min_count = 0x10000
    
    for i, (item_id, count) in enumerate(hh_entries):
        if item_id == item:
            new_entries.append((item_id, min(count + 1, 65535)))
            found = True
        else:
            new_entries.append((item_id, count))
            if count < min_count:
                min_count = count
                min_idx = i
    
    if not found:
        if hh_count < HH_SIZE:
            # Add new entry
            new_entries.append((item, 1))
            hh_count += 1
        else:
            # Replace the entry with minimum count
            if min_idx >= 0:
                new_entries[min_idx] = (item, 1)
    
    # Serialize CM back to bytes
    new_cm_data = bytearray(NUM_ROWS * NUM_COLS * 2)
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            offset = row * NUM_COLS * 2 + col * 2
            struct.pack_into('<H', new_cm_data, offset, cm[row][col])
    
    return _build_state(hh_count, new_entries, bytes(new_cm_data))

def query(state, item):
    """Query the estimated frequency of an item."""
    if len(state) != STATE_SIZE:
        return 0
    
    hh_count, hh_entries, cm_data = _parse_state(state)
    
    # Check deadline
    if time.monotonic() > _DEADLINE:
        return 0
    
    # Check if item is in heavy hitter table
    for item_id, count in hh_entries:
        if item_id == item:
            return float(count)
    
    # Otherwise, query CM sketch (take minimum across rows)
    min_est = 0x10000
    for row in range(NUM_ROWS):
        col = _hash_index(item, row)
        offset = row * NUM_COLS * 2 + col * 2
        val = struct.unpack_from('<H', cm_data, offset)[0]
        if val < min_est:
            min_est = val
    
    return float(min_est)
