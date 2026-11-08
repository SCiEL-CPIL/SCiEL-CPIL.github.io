def get_lattice_centers(n):
    centers = [(0.0, 0.0)]
    if n == 1:
        return [centers]
    
    ring1 = [
        (math.sqrt(3), 0),
        (math.sqrt(3)/2, 1.5),
        (-math.sqrt(3)/2, 1.5),
        (-math.sqrt(3), 0),
        (-math.sqrt(3)/2, -1.5),
        (math.sqrt(3)/2, -1.5)
    ]
    centers.extend(ring1)
    if n == 7:
        return [centers]
    
    ring2_axial = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
        (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1)
    ]
    ring2 = []
    for q, r in ring2_axial:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        ring2.append((x, y))
        
    needed = n - 7
    candidates = []
    
    # If needed <= 12, pick subsets of ring2
    # For efficiency and simplicity, we generate a few symmetric subsets
    if needed == 4:
        # Consecutive
        for i in range(12):
            subset = [ring2[(i+j) % 12] for j in range(4)]
            candidates.append(centers + subset)
        # Spaced by 3
        for i in range(4):
            subset = [ring2[(i+3*j) % 12] for j in range(4)]
            candidates.append(centers + subset)
        # Spaced by 2 (2 pairs of adjacent?)
        # Just add a couple of random ones if needed, but symmetric is better.
        
    # Fallback: just take the first 'needed'
    if not candidates:
        candidates.append(centers + ring2[:needed])
        
    return candidates
