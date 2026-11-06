"""Cap set priority function using pseudo-random permutation."""
import time
import hashlib

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s kill

def _get_priority(vector, n):
    """
    Compute priority for a vector in {0,1,2}^n.
    
    Strategy: Use a precomputed pseudo-random permutation of the entire space
    to assign globally unique and deterministic priorities. This avoids the
    local clustering of balance-based scores and explores a more uniform,
    non-greedy-friendly ordering of the hypercube.
    
    We encode the vector as a base-3 number and use a cryptographic hash
    to map it to a pseudo-random position in [0, 1). This creates a
    "random-looking" ordering that doesn't correlate with structural features.
    """
    # Encode vector as base-3 integer for deterministic indexing
    # This gives us a unique index for each vector in {0,1,2}^n
    base3_index = 0
    for i, v in enumerate(vector):
        base3_index = base3_index * 3 + v
    
    # Use SHA-256 for better distribution than MD5
    # Convert base3_index to bytes and hash it
    index_bytes = base3_index.to_bytes(max(1, (base3_index.bit_length() + 7) // 8), byteorder='big')
    h = hashlib.sha256(index_bytes).digest()
    
    # Take first 8 bytes to create a 64-bit integer for high precision
    hash_val = int.from_bytes(h[:8], byteorder='big')
    
    # Normalize to [0, 1)
    priority = hash_val / (2**64)
    
    return float(priority)

def priority(vector, n):
    """
    Priority function for cap set greedy construction.
    
    Args:
        vector: tuple of values in {0, 1, 2}
        n: length of the vector
    
    Returns:
        float: priority value (higher = added earlier)
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return a default priority to avoid hanging
        return 0.0
    
    try:
        return _get_priority(vector, n)
    except Exception:
        return 0.0
