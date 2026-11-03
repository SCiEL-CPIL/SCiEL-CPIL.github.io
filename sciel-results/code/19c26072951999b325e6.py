<<<<<<< SEARCH
    positions = []
    for q, r in coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        positions.append([x, y, 0.0]) # angle 0
=======
    # Scale positions to ensure strict disjointness (avoid touching/overlap edge cases)
    scale = 1.01
    positions = []
    for q, r in coords:
        x = 1.5 * q * scale
        y = np.sqrt(3) * (r + q / 2.0) * scale
        positions.append([x, y, 0.0]) # angle 0
>>>>>>> REPLACE
