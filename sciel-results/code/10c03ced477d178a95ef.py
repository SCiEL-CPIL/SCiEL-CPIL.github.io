def _luw_construct(k, g):
    # Copy of the seed logic
    q = k
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    d = (g - 1) // 2
    if d < 3: d = 3 # Ensure minimum depth for stability?
    # For g=9, d=4.
