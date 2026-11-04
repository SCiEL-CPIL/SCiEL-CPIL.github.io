<<<<<<< SEARCH
    # Add a tiny margin to the side length to ensure floating point safety
    # The evaluator checks if inner hexagons are strictly inside.
    # If we report exactly the minimum, floating point errors might cause a violation.
    side = best_score + 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)
=======
    # Add a tiny margin to the side length to ensure floating point safety
    # The evaluator checks if inner hexagons are strictly inside.
    # If we report exactly the minimum, floating point errors might cause a violation.
    side = best_score + 1e-10
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)
>>>>>>> REPLACE
