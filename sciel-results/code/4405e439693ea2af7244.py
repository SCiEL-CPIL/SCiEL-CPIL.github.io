<<<<<<< SEARCH
    # Convert to Cartesian
    centers = np.array([[sqrt3 * (q + r/2.0), 1.5 * r] for q, r in inner_positions_axial])
    
    # Center the arrangement at origin
    center_offset = centers.mean(axis=0)
    centers = centers - center_offset
=======
    # Convert to Cartesian
    centers = np.array([[sqrt3 * (q + r/2.0), 1.5 * r] for q, r in inner_positions_axial])
    
    # Scale slightly to ensure strict non-overlap due to floating point precision
    # Distance between neighbors is sqrt(3). FP error might make it slightly less.
    # Scaling by (1 + 1e-9) ensures d > sqrt(3).
    centers *= (1.0 + 1e-9)
    
    # Center the arrangement at origin
    center_offset = centers.mean(axis=0)
    centers = centers - center_offset
>>>>>>> REPLACE
