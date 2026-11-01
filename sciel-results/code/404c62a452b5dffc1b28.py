def luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Girth is 2d + 2.
    We need 2d + 2 >= g => d >= (g - 2) / 2.
    """
    q = k
    # Calculate minimal d
    d_min = (g - 2) // 2
    if (g - 2) % 2 != 0:
        d_min += 1
    d = max(3, d_min)
    
    # ... rest of code ...
