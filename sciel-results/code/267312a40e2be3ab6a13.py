def update(state, item):
    ba = bytearray(state)
    # 1. Update exact count
    c = ba[item]
    if c < 255:
        ba[item] = c + 1
        current_count = c + 1
    else:
        ba[item] = 255
        current_count = 255 # We will refine this in tracker if inserted

    # 2. Update Top-K
    # Top-K starts at 400
    # 16 slots, 4 bytes each
    # Slot i: [400 + 4*i, 400 + 4*i + 1] = item ID (LE uint16)
    # Slot i: [400 + 4*i + 2, 400 + 4*i + 3] = count (LE uint16)
    
    found = False
    min_idx = -1
    min_val = 65535
    
    for i in range(16):
        base = 400 + 4*i
        slot_id = ba[base] | (ba[base+1] << 8)
        slot_count = ba[base+2] | (ba[base+3] << 8)
        
        if slot_id == item:
            # Found item in tracker
            new_count = slot_count + 1
            ba[base+2] = new_count & 0xFF
            ba[base+3] = (new_count >> 8) & 0xFF
            found = True
            break
        
        if slot_count < min_val:
            min_val = slot_count
            min_idx = i

    if not found:
        # Check if we should insert
        # If current_count > min_val, replace
        # Note: if item was at 255, current_count is 255.
        # If min_val is 0, we insert.
        if current_count > min_val:
            base = 400 + 4*min_idx
            ba[base] = item & 0xFF
            ba[base+1] = (item >> 8) & 0xFF
            ba[base+2] = current_count & 0xFF
            ba[base+3] = (current_count >> 8) & 0xFF

    return bytes(ba)

def query(state, item):
    # Check Top-K first
    for i in range(16):
        base = 400 + 4*i
        slot_id = state[base] | (state[base+1] << 8)
        if slot_id == item:
            slot_count = state[base+2] | (state[base+3] << 8)
            return slot_count
    
    # Else return exact count
    return state[item]
