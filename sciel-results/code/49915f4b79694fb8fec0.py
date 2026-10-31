def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Maximize count of 1s (center of hypercube)
    # Minimize product of counts of 0s and 2s (boundary proximity)
    # This targets vectors near the "center" of the space where
    # affine lines are less dense
    
    center_score = c1
    
    # Penalize boundary vectors (those with many 0s or many 2s)
    # Use geometric mean to balance the two sides
    if c0 > 0 and c2 > 0:
        boundary_penalty = (c0 * c2) ** 0.5
    else:
        boundary_penalty = max(c0, c2)
    
    return center_score - 0.5 * boundary_penalty
