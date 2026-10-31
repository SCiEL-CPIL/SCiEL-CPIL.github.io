def priority(vector, n):
    # Hash-based priority with structural awareness
    # Mix coordinates with position-dependent weights to break symmetry
    h = 0
    for i in range(n):
        v = vector[i]
        # Position-dependent mixing
        h = (h ^ (v * (0x9E3779B9 | i))) & 0xFFFFFFFF
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    # Final avalanche
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = ((h << 15) | (h >> 17)) & 0xFFFFFFFF
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    return h
