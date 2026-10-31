def priority(vector, n):
    s = 0
    val = 0
    for v in vector:
        if v == 1:
            s += 1
        elif v == 2:
            s -= 1
        val = val * 3 + v
        
    # Prioritize balanced vectors
    # s=0 is best, then s=1/-1, etc.
    # Use -s^2 as primary score.
    # Add small tie-breaker based on val to ensure unique ordering.
    # val is in [0, 3^n - 1].
    # To avoid overflow or precision issues, we can scale val.
    # But for small n, it's fine.
    # For large n, val is huge.
    # We can use val / (3**n) as a fraction in [0, 1).
    # But 3**n is expensive to compute repeatedly.
    # We can just use a small epsilon.
    
    # If n is large, s^2 is small compared to val.
    # So the tie-breaker might dominate if epsilon is not small enough.
    # But s^2 is integer.
    # So if we use epsilon = 1e-6, and val < 1e6, it's fine.
    # For n=10, val < 60000.
    # For n=15, val < 14 million.
    # 1e-6 * 14e6 = 14.
    # This would dominate s^2 (max 225).
    # So we need a smaller epsilon for larger n.
    # Or normalize val.
    
    # Let's use a normalized tie-breaker.
    # max_val = 3**n - 1
    # norm_val = val / max_val
    # priority = -s^2 - 1e-6 * norm_val
    
    # But 3**n is expensive.
    # We can compute it once? No, priority is called per vector.
    # We can pass n.
    # 3**n is fast for n < 20.
    
    max_val = 3**n - 1
    if max_val == 0:
        norm_val = 0.0
    else:
        norm_val = val / max_val
        
    return -s*s - 1e-6 * norm_val
