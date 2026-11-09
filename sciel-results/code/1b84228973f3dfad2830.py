import struct

# Constants
W = 512          # width of each row (512 counters)
D = 2            # depth (number of rows)
PRIME = 2147483647

# Pre-compute hash parameters
# We use the same primes as the seed for consistency, 
# but the width W is now 512.
_PS = (1035493289, 775374203) # Only need 2 primes for D=2

# Pre-compute indices for all items (0..399) and all rows (0..1)
# _INDICES[item][row] = ((item * p + 31 * row) % PRIME) % W
_INDICES = [[0] * D for _ in range(400)]
for item in range(400):
    for r in range(D):
        p = _PS[r]
        _INDICES[item][r] = ((item * p + 31 * r) % PRIME) % W

def update(state, item):
    """
    Update the Count-Min sketch state.
    state: bytes of length 1024 (2 rows * 512 counters * 1 byte per counter)
    item: int in 0..399
    returns: new state as bytes
    """
    # Convert state to a mutable buffer
    buf = bytearray(state)
    
    # Get precomputed indices for this item
    idxs = _INDICES[item]
    
    # For each row, increment the counter at the precomputed index
    for r in range(D):
        offset = r * W + idxs[r]
        
        # Read current value
        current = buf[offset]
        
        # Increment (saturating at 255)
        if current < 255:
            buf[offset] = current + 1
        # else: stays 255
        
    return bytes(buf)

def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate (int)
    """
    # Use the minimum of the D row estimates
    min_est = 256  # Start higher than max uint8
    
    # Get precomputed indices for this item
    idxs = _INDICES[item]
    
    for r in range(D):
        offset = r * W + idxs[r]
        
        # Read the counter value
        val = state[offset]
        
        if val < min_est:
            min_est = val
            
    return int(min_est)
