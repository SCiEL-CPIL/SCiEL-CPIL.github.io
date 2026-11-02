<<<<<<< SEARCH
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
=======
    q = k
    # LZW graph girth is at least 2d+1.
    # We need 2d+1 >= g => d >= (g-1)/2.
    # So d = ceil((g-1)/2) = g//2 for integer g.
    d = max(2, g // 2)
>>>>>>> REPLACE
