import hashlib

def priority(vector, n):
    # Calculate Hamming weight (number of non-zero elements)
    weight = 0
    for v in vector:
        if v != 0:
            weight += 1
            
    # We want low weight to have high priority.
    # So we negate the weight.
    # To break ties deterministically but pseudo-randomly, we use a hash.
    # This avoids the lexicographic bias of the seed.
    
    # Convert vector to string for hashing
    # Using a simple hash is faster than hashlib for small n, but hashlib is safer for uniformity
    # Since n is small, a simple integer hash is fine.
    h = 0
    for v in vector:
        h = (h * 31 + v) & 0xFFFFFFFF
        
    # Normalize hash to [0, 1)
    noise = (h % 1000000) / 1000000.0
    
    # Primary key: -weight
    # Secondary key: noise
    # We scale noise so it doesn't override weight
    return -weight + noise
