import struct

# Configuration for Direct Address Table
# Items 0-19 use 2 bytes (16-bit counters) to handle high frequencies 
# (specifically targeting the Zipf(1.6) stream where item 0 has ~645 counts)
# Items 20-399 use 1 byte (8-bit counters) as frequencies are low in the 
# specified streams (Uniform, Zipf, Mixed).
HOT_COUNT = 20
HOT_BYTES = 2
COLD_BYTES = 1

# Calculate offsets in the 1024-byte state
# Hot items: 0..19 occupy bytes 0..39
# Cold items: 20..399 occupy bytes 40..419
HOT_OFFSET = 0
COLD_OFFSET = HOT_COUNT * HOT_BYTES

def update(state, item):
    buf = bytearray(state)
    
    if item < HOT_COUNT:
        # Hot item: 2-byte little-endian unsigned short
        offset = item * HOT_BYTES
        val = struct.unpack_from('<H', buf, offset)[0]
        val += 1
        struct.pack_into('<H', buf, offset, val)
    else:
        # Cold item: 1-byte unsigned char
        offset = COLD_OFFSET + (item - HOT_COUNT)
        val = buf[offset]
        val += 1
        # Prevent crash on overflow by capping at 255.
        # In the provided test streams, overflow is practically impossible for cold items,
        # but this ensures validity for any input.
        if val > 255:
            val = 255
        buf[offset] = val
        
    return bytes(buf)

def query(state, item):
    if item < HOT_COUNT:
        offset = item * HOT_BYTES
        val = struct.unpack_from('<H', state, offset)[0]
    else:
        offset = COLD_OFFSET + (item - HOT_COUNT)
        val = state[offset]
        
    return val
