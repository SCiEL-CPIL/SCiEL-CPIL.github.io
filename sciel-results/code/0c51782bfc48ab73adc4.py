def construct(k, g):
    q = k
    # Ensure d is at least 4 for girth 9 with q=5
    # Girth of CD(d,q) is d+5.
    # We need d+5 >= g => d >= g-5.
    # Also d must be >= 3 for the construction to be valid in the code's logic.
    d = max(3, g - 5)
    
    # ... rest of code ...
