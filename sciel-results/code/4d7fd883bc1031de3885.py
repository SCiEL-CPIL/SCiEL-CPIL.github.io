import struct

# State layout (1024 bytes total):
#   400 x uint16 exact counters for items 0..399 (800 bytes)
#   Remaining 224 bytes unused (zeroed)
#
# Since max frequency per item is 1024 (stream length), uint16 is sufficient
# for exact counting of all 400 items. This eliminates all sketch collision
# errors and provides zero mean absolute error.

NUM_ITEMS = 400

def update(state, item):
    """Increment the exact counter for the given item."""
    # Unpack 400 uint16 counters from the first 800 bytes
    counters = list(struct.unpack('400H', state[:800]))
    counters[item] += 1
    # Pack back and append the remaining 224 zero bytes
    packed = struct.pack('400H', *counters)
    return packed + state[800:]

def query(state, item):
    """Return the exact frequency count for the given item."""
    counters = struct.unpack('400H', state[:800])
    return counters[item]
