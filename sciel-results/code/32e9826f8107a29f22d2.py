def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    ideal = n / 3.0
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Compute a deterministic hash for tie-breaking.
    # We want a value in [0, 1) that is well-distributed.
    # Using a simple polynomial hash with a large prime base.
    # We convert the vector to a string or use the tuple directly?
    # Tuple hash in Python is deterministic for the same object, but we want
    # a stable, custom hash to ensure specific distribution properties.
    
    # FNV-1a style hash for speed and determinism
    hash_val = 0xcbf29ce484222325
    prime = 0x100000001b3
    
    for v in vector:
        hash_val ^= v
        hash_val = (hash_val * prime) & 0xFFFFFFFFFFFFFFFF
    
    # Normalize to [0, 1)
    # We take the high bits to avoid modulo bias if any, though for a float it's fine.
    # We want the hash to act as a secondary sort key.
    # Since we are returning a float to be sorted in descending order,
    # we want the hash part to be positive and small so it doesn't overwhelm 'dev'.
    # But 'dev' is the primary key.
    # If we return -dev - hash_norm, vectors with lower dev are preferred.
    # Among those with same dev, lower hash_norm is preferred (since we subtract).
    # This creates a deterministic, scattered order among balanced vectors.
    
    # To make the hash contribution negligible compared to the smallest possible difference in 'dev',
    # we scale it down.
    # Min non-zero difference in 'dev' depends on n.
    # For n divisible by 3, dev can be 0, 2, 8, etc.
    # For n not divisible by 3, dev is non-integer.
    # Let's just use a very small weight, e.g., 1e-6.
    # But wait, if we use 1e-6, and hash is in [0,1), the difference is < 1e-6.
    # This is effectively a tie-breaker.
    
    # However, the directive suggests a "static, pseudo-random permutation".
    # A simple hash provides this.
    
    hash_norm = hash_val / 0xFFFFFFFFFFFFFFFF
    return -dev - hash_norm * 1e-6
