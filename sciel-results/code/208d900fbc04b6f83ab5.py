    # LUW graph CD(d, q) has girth 2d + 1
    # We need 2d + 1 >= g  =>  d >= (g - 1) / 2
    d = (g - 1 + 1) // 2  # ceil((g-1)/2)
    d = max(2, d)
    