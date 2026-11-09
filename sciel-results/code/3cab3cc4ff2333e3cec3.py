<<<<<<< SEARCH
    # LUW bound: girth >= d + 5 for odd d
    # For girth g, need d >= g - 5
    min_d = max(3, g - 5)
    if min_d % 2 == 0:
        min_d += 1  # Ensure odd for the girth guarantee
=======
    # LUW bound: girth >= 2d + 1
    # For girth g, need 2d + 1 >= g => d >= (g - 1) / 2
    min_d = max(2, (g - 1) // 2)
>>>>>>> REPLACE
