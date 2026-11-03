# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns 13 points: 1 at the center, 12 on a regular 12-gon ring.
    The score is invariant to the radius and phase of the regular ring,
    so we simply return a standard configuration.
    """
    n_outer = 12
    radius = 1.0
    phase = 0.0
    
    # Center point
    center = np.array([0.0, 0.0])
    
    # Outer ring points
    angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + phase
    outer_pts = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    
    # Combine
    points = np.vstack([center, outer_pts])
    
    return points
# EVOLVE-BLOCK-END
