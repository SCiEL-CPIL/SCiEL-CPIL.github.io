import struct

def update(state, item):
    # State layout:
    # Bytes 0-127: 1024-bit bitset for items 0..1023 (we use 0..399)
    # Bytes 128-131: Total count (uint32 LE)
    # Bytes 132-135: Distinct count (uint32 LE)
    # Rest: 0
    
    # Copy state to new bytearray
    new_state = bytearray(state)
    
    # Extract counters
    total_count = struct.unpack('<I', new_state[128:132])[0]
    distinct_count = struct.unpack('<I', new_state[132:136])[0]
    
    # Update bitset
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    # Check if item is already seen
    if not (new_state[byte_idx] & mask):
        new_state[byte_idx] |= mask
        distinct_count += 1
    
    # Increment total count
    total_count += 1
    
    # Update counters in new_state
    new_state[128:132] = struct.pack('<I', total_count)
    new_state[132:136] = struct.pack('<I', distinct_count)
    
    return bytes(new_state)

def query(state, item):
    # Parse state
    total_count = struct.unpack('<I', state[128:132])[0]
    distinct_count = struct.unpack('<I', state[132:136])[0]
    
    # Check membership
    byte_idx = item // 8
    bit_idx = item % 8
    mask = 1 << bit_idx
    
    if not (state[byte_idx] & mask):
        return 0.0
    
    if distinct_count == 0:
        return 0.0
        
    return float(total_count) / float(distinct_count)
