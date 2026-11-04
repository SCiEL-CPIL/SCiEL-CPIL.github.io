<<<<<<< SEARCH
    q = k
    # For LUW: girth >= d + 5 if d is odd, d + 6 if d is even
    # We need girth >= g
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
=======
    q = k
    # Find minimal d such that LUW girth bound >= g
    # Bound: d + 5 if d is odd, d + 6 if d is even
    d = 3
    while True:
        if d % 2 == 1:
            girth_bound = d + 5
        else:
            girth_bound = d + 6
        if girth_bound >= g:
            break
        d += 1
    d = max(3, d)
>>>>>>> REPLACE
