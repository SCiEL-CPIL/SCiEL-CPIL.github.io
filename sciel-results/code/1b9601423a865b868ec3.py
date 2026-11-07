"""Memory-bounded streaming frequency sketch using direct 8-bit counters + overflow tracking."""
import struct

# State layout: 1024 bytes total
# Bytes 0..399: 8-bit counters for items 0..399
# Bytes 400..1023: 624 bytes for overflow tracking (312 2-bit flags, but we'll use a simpler scheme)

# Actually, let's think about this more carefully.
# 400 items, 1024 total insertions per stream.
# Max count for any single item: if all 1024 go to one item, count = 1024.
# But with 400 possible items, the max is at most 1024.
# 8-bit counter max is 255. So we need to handle overflow.

# Strategy: 
# - Bytes 0..399: 8-bit counters (0-255)
# - For overflow: we need to track items that exceed 255.
# - With 1024 items and 400 values, the max count is 1024 (if all go to one item).
# - 1024 needs 10 bits. So we could use 2 extra bits per item? That's 400*2 = 800 bits = 100 bytes.
# - Total: 400 + 100 = 500 bytes. Well within 1024.

# Better plan:
# Bytes 0..399: 8-bit base counters
# Bytes 400..499: 100 bytes = 800 bits, 2 bits per item for overflow (values 256-1023 need 10 bits total, so 2 extra bits give 4*256=1024 range)
# Actually 2 extra bits means counter can go up to 4*256 = 1024. Perfect.

# Let me reconsider: 8 bits gives 0-255. With 2 more bits, we can represent 0-1023 (10 bits total).
# But we only have 8 bits stored directly. So:
# - Store low 8 bits in bytes 0..399
# - Store high 2 bits (bits 8 and 9) in a bitfield

# Bitfield: 400 items * 2 bits = 800 bits = 100 bytes
# Bytes 400..499: bitfield where for item i, the 2 bits are at position (i*2) and (i*2+1)
# Bit 0 of the pair = bit 8 of count, bit 1 of the pair = bit 9 of count

def update(state, item):
    """Update the sketch state with a new item."""
    state = bytearray(state)
    
    # Get current count for this item
    base = state[item]  # low 8 bits
    # Get high 2 bits from bitfield
    bit_offset = item * 2
    byte_idx = 400 + (bit_offset // 8)
    bit_pos = bit_offset % 8
    high_bits_byte = state[byte_idx]
    
    # Extract 2 high bits
    high = (high_bits_byte >> bit_pos) & 0x03
    count = base | (high << 8)
    
    # Increment
    count += 1
    
    # If count exceeds 1023, cap it (shouldn't happen with 1024 items)
    if count > 1023:
        count = 1023
    
    # Update low 8 bits
    state[item] = count & 0xFF
    
    # Update high 2 bits
    new_high = (count >> 8) & 0x03
    # Clear old high bits
    mask = ~(0x03 << bit_pos) & 0xFF
    high_bits_byte = (high_bits_byte & mask) | (new_high << bit_pos)
    state[byte_idx] = high_bits_byte
    
    return bytes(state)

def query(state, item):
    """Query the estimated frequency of an item."""
    # Get low 8 bits
    base = state[item]
    
    # Get high 2 bits from bitfield
    bit_offset = item * 2
    byte_idx = 400 + (bit_offset // 8)
    bit_pos = bit_offset % 8
    high_bits_byte = state[byte_idx]
    
    # Extract 2 high bits
    high = (high_bits_byte >> bit_pos) & 0x03
    count = base | (high << 8)
    
    return count
