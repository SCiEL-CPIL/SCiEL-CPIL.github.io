<<<<<<< SEARCH
    # Construct the inner list
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
    
    return inner, (0.0, 0.0), S, 0.0
=======
    # Construct the inner list
    # Rotate inner hexagons by 30 degrees to fit the lattice spacing
    # Angle 0 corresponds to flat-topped (horizontal top/bottom), width 2.
    # Angle 30 corresponds to pointy-topped (vertical sides), width sqrt(3).
    # The lattice uses x-spacing sqrt(3), so we need width sqrt(3).
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 30.0])
    
    return inner, (0.0, 0.0), S, 0.0
>>>>>>> REPLACE
