import hashlib
import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a deterministic hash to randomize the order of addition.
    # This avoids the clustering effect of lexicographic order.
    # We convert the tuple to a string or bytes and hash it.
    # Using a simple integer conversion is faster.
    
    # Convert vector to a single integer for hashing
    # Since elements are 0, 1, 2, we can encode it in base 3.
    val = 0
    for i in range(n):
        val = val * 3 + vector[i]
    
    # Use the integer value itself or a hash of it.
    # Simply returning val would be lexicographic (if n is fixed and we read from end).
    # We want to shuffle it.
    
    # A simple multiplicative hash
    h = (val * 2654435761) & 0xFFFFFFFF
    return float(h)
# EVOLVE-BLOCK-END
