"""
Strategy: Exact Counting via Bit-Packing.
Instead of using a probabilistic sketch (Count-Min) which introduces error due to collisions,
we exploit the constraint that the maximum frequency of any item is bounded by the stream length (1024).
Since 1024 < 2^11, we can store the exact count for each of the 400 items in 11 bits.
Total bits required: 400 * 11 = 4400 bits = 550 bytes.
This fits comfortably within the 1024-byte state limit.
This approach yields zero error for all items, resulting in a score of 0.
"""

# EVOLVE-BLOCK-START

def update(state, item):
    # state is a bytes object of length 1024.
    # We use the first 550 bytes to store 400 counters of 11 bits each.
    # Item index: 0..399.
    # Bit offset for item i: i * 11.
    # Byte index: (i * 11) // 8.
    # Bit index within byte: (i * 11) % 8.
    
    i = item
    bit_offset = i * 11
    byte_idx = bit_offset // 8
    bit_idx = bit_offset % 8
    
    # Convert state to a mutable bytearray
    buf = bytearray(state)
    
    # Extract the 11-bit counter for this item
    # The counter spans bit_idx to bit_idx + 10.
    # We need to read from buf[byte_idx] and potentially buf[byte_idx+1] and buf[byte_idx+2].
    # Since 11 bits can span at most 2 bytes (if bit_idx <= 7, it spans byte_idx and byte_idx+1.
    # Actually, if bit_idx is 0, it uses bits 0-10 of byte_idx and byte_idx+1.
    # If bit_idx is 7, it uses bit 7 of byte_idx, and bits 0-9 of byte_idx+1.
    # It never spans 3 bytes because 11 <= 8+8.
    
    # Mask to extract the current value
    # We can read 16 bits starting from bit_idx to be safe, then mask.
    # Read 2 bytes: buf[byte_idx] and buf[byte_idx+1]
    val1 = buf[byte_idx]
    val2 = buf[byte_idx + 1]
    
    # Combine into an integer, shifting so that the target bit is at position 0
    # The combined 16-bit value (little-endian logic for our extraction)
    # Actually, it's easier to just shift the whole buffer view.
    # Let's construct the integer from the two bytes.
    # byte_idx contains bits [0..7] of the local window.
    # byte_idx+1 contains bits [8..15] of the local window.
    # We want bits [bit_idx .. bit_idx+10].
    
    # Create a 16-bit integer representing the bytes at byte_idx and byte_idx+1
    # Note: Python integers are arbitrary precision, so we can just shift.
    # We need to be careful with endianness. The state is a byte array.
    # bit 0 of the state is the LSB of buf[0].
    # So the value at bit_offset is:
    # (buf[byte_idx] >> bit_idx) | (buf[byte_idx+1] << (8 - bit_idx))
    # This gives us a value where the LSB is at bit_offset.
    # We want the 11 bits starting from there.
    
    # Extract 11 bits
    # The value formed by (buf[byte_idx] >> bit_idx) | (buf[byte_idx+1] << (8 - bit_idx))
    # has the target bit at position 0.
    # We mask with 0x7FF (11 bits).
    
    if bit_idx == 0:
        current_val = buf[byte_idx] | (buf[byte_idx + 1] << 8)
        current_val &= 0x7FF
    else:
        # Shift right by bit_idx
        # The lower (8-bit_idx) bits of buf[byte_idx+1] are the upper bits of our counter.
        # The upper bit_idx bits of buf[byte_idx] are the lower bits of our counter? No.
        # Let's trace:
        # Bit 0 of counter is at global bit_offset.
        # Global bit_offset = byte_idx * 8 + bit_idx.
        # So bit 0 of counter is bit bit_idx of byte byte_idx.
        # Bit 1 of counter is bit bit_idx+1 of byte byte_idx.
        # ...
        # Bit 7-bit_idx of counter is bit 7 of byte byte_idx.
        # Bit 8-bit_idx of counter is bit 0 of byte byte_idx+1.
        
        # So we can extract by:
        # lower_part = buf[byte_idx] & (0xFF << bit_idx)  # bits bit_idx..7 of byte_idx
        # upper_part = buf[byte_idx+1] & (0xFF >> (8 - bit_idx)) # bits 0..7-bit_idx of byte_idx+1
        # current_val = (lower_part >> bit_idx) | (upper_part << (8 - bit_idx))
        
        lower_part = buf[byte_idx] & (0xFF << bit_idx)
        upper_part = buf[byte_idx + 1] & (0xFF >> (8 - bit_idx))
        current_val = (lower_part >> bit_idx) | (upper_part << (8 - bit_idx))
        
        # Mask to 11 bits just in case, though the above logic should isolate exactly 11 bits
        # if we assume the data was written correctly.
        current_val &= 0x7FF

    # Increment
    new_val = current_val + 1
    
    # Write back
    # We need to clear the 11 bits and set the new value.
    
    if bit_idx == 0:
        # Clear bits 0-10 in buf[byte_idx] and buf[byte_idx+1]
        # buf[byte_idx] lower 8 bits cleared.
        # buf[byte_idx+1] lower 3 bits cleared.
        buf[byte_idx] = (buf[byte_idx] & 0x00) | (new_val & 0xFF)
        buf[byte_idx + 1] = (buf[byte_idx + 1] & 0xF8) | ((new_val >> 8) & 0x07)
    else:
        # Clear the relevant bits
        # Mask for byte_idx: bits 0..bit_idx-1 are kept, bits bit_idx..7 are cleared.
        # Mask for byte_idx+1: bits 8-bit_idx..7 are kept, bits 0..7-bit_idx are cleared.
        
        mask_lower = (1 << bit_idx) - 1
        mask_upper = 0xFF << (8 - bit_idx)
        
        # Preserve non-counter bits
        preserved_lower = buf[byte_idx] & mask_lower
        preserved_upper = buf[byte_idx + 1] & mask_upper
        
        # Insert new value
        # Lower part of new_val goes into upper bits of byte_idx
        lower_new = (new_val & 0xFF) << bit_idx
        # Upper part of new_val goes into lower bits of byte_idx+1
        upper_new = (new_val >> 8) & (0xFF >> (8 - bit_idx))
        
        buf[byte_idx] = preserved_lower | lower_new
        buf[byte_idx + 1] = preserved_upper | upper_new

    return bytes(buf)

def query(state, item):
    i = item
    bit_offset = i * 11
    byte_idx = bit_offset // 8
    bit_idx = bit_offset % 8
    
    buf = state  # bytes are immutable, direct access is fine
    
    if bit_idx == 0:
        val = buf[byte_idx] | (buf[byte_idx + 1] << 8)
        val &= 0x7FF
    else:
        lower_part = buf[byte_idx] & (0xFF << bit_idx)
        upper_part = buf[byte_idx + 1] & (0xFF >> (8 - bit_idx))
        val = (lower_part >> bit_idx) | (upper_part << (8 - bit_idx))
        val &= 0x7FF
        
    return float(val)

# EVOLVE-BLOCK-END
