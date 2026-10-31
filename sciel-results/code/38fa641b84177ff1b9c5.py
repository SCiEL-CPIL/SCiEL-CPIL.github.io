def priority(vector, n):
    # Convert vector to a base-3 integer
    val = 0
    for i, v in enumerate(vector):
        val = val * 3 + v
    # Use a simple hash to spread the values
    # Python's built-in hash is not stable across runs due to hash randomization.
    # We need a deterministic hash.
    # We can use a simple LCG or xorshift based on val.
    # Or just use val itself? No, val is just the lexicographic index.
    # We need to permute the indices.
    
    # Let's use a simple deterministic pseudo-random permutation.
    # e.g., val * large_prime % modulus
    # But we need it to be in [0, 1) or just a float.
    
    # A good simple hash:
    h = val
    h ^= (h >> 16)
    h *= 0x85ebca6b
    h ^= (h >> 13)
    h *= 0xc2b2ae35
    h ^= (h >> 16)
    return h / 2**32
