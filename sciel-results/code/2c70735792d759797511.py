import hashlib
from collections import Counter

def priority(vector, n):
    # Use Counter for efficient counting
    counts = Counter(vector)
    c0 = counts.get(0, 0)
    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    
    # Calculate deviation from ideal balance (n/3)
    ideal = n / 3.0
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Generate a deterministic hash for tie-breaking
    # Using md5 of the vector string representation
    # This provides a high-quality, deterministic tie-breaker
    # that is structurally distinct from simple polynomial loops
    vec_str = ','.join(map(str, vector))
    h_digest = hashlib.md5(vec_str.encode()).digest()
    
    # Convert first 8 bytes to an integer for a 64-bit hash value
    # This ensures a wide range of tie-breaking values
    h_val = int.from_bytes(h_digest[:8], byteorder='big')
    
    # Normalize the hash to a small range [0, 1) to act as a tie-breaker
    # The primary objective is minimizing dev (maximizing -dev)
    # The hash breaks ties in a deterministic, pseudo-random manner
    norm_h = h_val / (2**64)
    
    # Return negative deviation (to maximize) minus a tiny fraction of the hash
    # The hash contribution is negligible compared to dev differences > 1e-6
    # but sufficient to break exact ties in dev
    return -dev - norm_h * 1e-6
