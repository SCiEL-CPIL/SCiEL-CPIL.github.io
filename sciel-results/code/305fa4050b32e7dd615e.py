# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Generates a path for the moving sofa problem using a circular arc model.
    The inner corner of the L-shaped corridor moves along a circular arc
    centered at (-1, 1) with radius sqrt(2), from (0, 0) to (-2, 0).
    This path dips downwards, potentially preserving more area in the lower
    half-plane where the start and end constraints are most restrictive.
    """
    # Parameters for the circular arc
    center_x = -1.0
    center_y = 1.0
    radius = np.sqrt(2.0)
    
    # Start and end angles in radians
    # Start point (0,0) relative to center (-1,1) is (1,-1), angle -45 deg
    # End point (-2,0) relative to center (-1,1) is (-1,-1), angle -135 deg
    start_angle = -np.pi / 4.0
    end_angle = -3.0 * np.pi / 4.0
    
    # Number of intermediate points to generate
    # We exclude the start and end points as they are added by the evaluator.
    # Using 20 points provides a good approximation of the arc.
    n_points = 20
    
    # Generate angles strictly between start and end
    # Note: start > end in value, so we step downwards
    angles = np.linspace(start_angle, end_angle, n_points + 2)[1:-1]
    
    # Calculate positions
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)
    
    # Calculate rotation angles in degrees
    # a = 0 at start, a = 90 at end
    # Linear interpolation of a with respect to the arc parameter
    # Since we are moving from -45 to -135, the fraction of the arc is (angle - start) / (end - start)
    # But we want a to go 0 -> 90.
    # Let t be 0 at start, 1 at end.
    # angles are linear in t.
    # a = 90 * t
    t = (angles - start_angle) / (end_angle - start_angle)
    a_coords = t * 90.0
    
    # Construct the path array
    path = np.column_stack((x_coords, y_coords, a_coords))
    
    return path


# EVOLVE-BLOCK-END
