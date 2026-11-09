def priority(vector, n):
    # Calculate distance from all-ones and all-twos vectors
    # This penalizes "low frequency" vectors that are highly correlated
    # with the all-ones direction, which tend to form 3-term APs.
    d1 = sum((x - 1) ** 2 for x in vector)
    d2 = sum((x - 2) ** 2 for x in vector)
    
    # Add a small tie-breaker to ensure determinism without bias
    h = hash(vector)
    tie_breaker = (h % 1000000) / 1000000.0
    
    # Prioritize vectors far from the low-frequency directions
    return d1 + d2 + tie_breaker
