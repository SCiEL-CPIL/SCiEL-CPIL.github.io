def construct_lw(k, g, d=None):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction."""
    q = k
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # ... rest of code
