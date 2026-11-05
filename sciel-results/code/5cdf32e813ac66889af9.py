<<<<<<< SEARCH
def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    This is a known good configuration.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # The side length for the optimal flower packing is exactly 4.0
    # Add a tiny margin for floating point safety
    side = 4.0 + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0
=======
def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    This is a known good configuration.
    """
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # The side length for the optimal flower packing is exactly 4.0
    # Add a tiny margin for floating point safety
    side = 4.0 + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0
>>>>>>> REPLACE
