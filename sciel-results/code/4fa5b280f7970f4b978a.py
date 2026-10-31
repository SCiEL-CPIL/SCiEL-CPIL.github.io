def priority(vector, n):
    # Convert vector to a base-3 integer
    val = 0
    weight = 0
    for i, v in enumerate(vector):
        val = val * 3 + v
        if v != 0:
            weight += 1
            
    # Penalize zero vector heavily to allow pairs {v, 2v}
    if val == 0:
        return 0.0

    # Use a simple hash to spread the values
    h = val
    h ^= (h >> 16)
    h *= 0x85ebca6b
    h ^= (h >> 13)
    h *= 0xc2b2ae35
    h ^= (h >> 16)
    base_priority = h / 2**32
    
    # Add a small bonus for higher weight to prefer "generic" vectors
    # This helps avoid sparse vectors which are in dense subspaces with many APs
    bonus = (weight / n) * 0.1
    
    return base_priority + bonus
