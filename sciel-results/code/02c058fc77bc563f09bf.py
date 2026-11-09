# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Generates a path for the moving sofa problem.
    Strategy: Delay the translation of the inner corner.
    Keep the corner at (0,0) for the first 45 degrees of rotation.
    Then move linearly to (-2,0) while rotating from 45 to 90 degrees.
    This keeps the "hole" of the L-shape away from the upper-right part of the sofa
    for longer, potentially preserving more area in the intersection.
    """
    n_points = 20
    
    # We generate points strictly between (0,0,0) and (-2,0,90).
    # The evaluator adds the endpoints.
    # We split the angle range [0, 90] into two phases.
    # Phase 1: a in [0, 45]. x=0, y=0.
    # Phase 2: a in [45, 90]. x moves 0 to -2, y=0.
    
    # We need n_points intermediate points.
    # Let's distribute them evenly in angle.
    angles = np.linspace(0, 90, n_points + 2)[1:-1]
    
    x_coords = np.zeros(n_points)
    y_coords = np.zeros(n_points)
    a_coords = angles
    
    for i in range(n_points):
        a = angles[i]
        if a <= 45.0:
            x_coords[i] = 0.0
            y_coords[i] = 0.0
        else:
            # Linear interpolation from 45 to 90
            t = (a - 45.0) / 45.0
            x_coords[i] = 0.0 + t * (-2.0 - 0.0)
            y_coords[i] = 0.0
    
    path = np.column_stack((x_coords, y_coords, a_coords))
    return path


# EVOLVE-BLOCK-END
