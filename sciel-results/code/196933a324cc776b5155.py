"""Memory-bounded streaming frequency sketches: Fixed-Window Exact + CM tail (uint16)."""
# EVOLVE-BLOCK-START
import struct
import array

# Layout:
# Bytes 0..255: exact counters for items 0..255 (1 byte each, capped at 255)
# Bytes 256..1023: 4 rows x 96 slots x uint16 = 4*96*2 = 768 bytes.

SKETCH_ROWS = 4
SKETCH_COLS = 96
SKETCH_OFFSET = 256
SKETCH_STRIDE = SKETCH_COLS * 2  # 192 bytes per row (96 uint16s)

# Hash constants for the tail sketch (items 256..399)
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)

def update(state, item):
    """
    Update the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: new state as bytes (1024 bytes)
    """
    if not isinstance(state, bytes) or len(state) != 1024:
        return b'\x00' * 1024
    
    buf = bytearray(state)
    
    if item < 256:
        # Exact counter
        idx = item
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
    else:
        # Tail item: use CM sketch with uint16 counters
        for row in range(SKETCH_ROWS):
            p = PRIMES[row]
            h = ((item * p + 31 * row) % 2147483647) % SKETCH_COLS
            
            # uint16 little-endian at offset
            offset = SKETCH_OFFSET + row * SKETCH_STRIDE + h * 2
            
            val = buf[offset] | (buf[offset + 1] << 8)
            
            if val < 65535:
                val += 1
            buf[offset] = val & 0xFF
            buf[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes (1024 bytes)
    item: int (0..399)
    returns: nonnegative frequency estimate (int or float)
    """
    if not isinstance(state, bytes) or len(state) != 1024:
        return 0
    
    if item < 256:
        return state[item]
    else:
        min_val = 65535
        for row in range(SKETCH_ROWS):
            p = PRIMES[row]
            h = ((item * p + 31 * row) % 2147483647) % SKETCH_COLS
            
            offset = SKETCH_OFFSET + row * SKETCH_STRIDE + h * 2
            
            val = state[offset] | (state[offset + 1] << 8)
            
            if val < min_val:
                min_val = val
        
        return min_val
# EVOLVE-BLOCK-END
