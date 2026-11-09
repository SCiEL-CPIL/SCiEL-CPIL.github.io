<<<<<<< SEARCH
    points_cart = []
    for q, r in points_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        points_cart.append((x, y))
        
    points_cart = np.array(points_cart)
=======
    points_cart = []
    for q, r in points_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        points_cart.append((x, y))
        
    points_cart = np.array(points_cart)
    # Scale up slightly to ensure strict non-overlap (disjoint interiors)
    # and avoid floating point precision issues with touching boundaries.
    points_cart = points_cart * 1.001
>>>>>>> REPLACE
