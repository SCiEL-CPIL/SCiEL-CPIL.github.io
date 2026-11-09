def priority(vector, n):
    # Calculate squared distances to the three constant vectors:
    # 0-vector: (0,0,...,0)
    # 1-vector: (1,1,...,1)
    # 2-vector: (2,2,...,2)
    
    d0 = sum(x * x for x in vector)
    d1 = sum((x - 1) ** 2 for x in vector)
    d2 = sum((x - 2) ** 2 for x in vector)
    
    # Sum of distances penalizes vectors close to any constant vector?
    # No, we want to PENALIZE high correlation to all-ones (1-vector).
    # The directive says "penalizing vectors with high correlation to the all-ones vector".
    # This implies we want LOW priority for vectors close to (1,1,...,1).
    # So we want HIGH priority for vectors FAR from (1,1,...,1).
    # d1 measures distance to 1-vector.
    # If we use just d1, we prefer 0s and 2s equally.
    
    # Let's stick to d1 as the primary term.
    # To break ties and add "high frequency" preference (balance),
    # we can add a small term that rewards variance?
    # Or just use d1.
    
    # The current program uses d1 + d2 which breaks symmetry.
    # Let's try just d1.
    
    # Add a tie breaker for determinism.
    h = hash(vector)
    tie_breaker = (h % 1000000) / 1000000.0
    
    return d1 + tie_breaker
