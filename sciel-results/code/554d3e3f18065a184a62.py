# EVOLVE-BLOCK-START
import numpy as np


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Erdos-Szekeres "Happy Ending" problem. Place n = 33 points in general
    position (no three collinear) inside the unit square so as to MINIMIZE the number of
    convex 7-gons whose vertices are among the points. Here k = 7, so n = 2^5 + 1 = 33.
    
    Returns:
        points: np.ndarray of shape (33, 2) with the x,y coordinates of the points,
            each lying in the unit square [0, 1] x [0, 1].
    """
    n = 33
    np.random.seed(42)
    
    # Two concave arcs construction
    n_upper = 17
    n_lower = 16
    
    # Upper arc: concave down
    x_upper = np.linspace(0.05, 0.95, n_upper)
    y_upper = 0.9 - 0.8 * x_upper**2
    
    # Lower arc: concave up  
    x_lower = np.linspace(0.05, 0.95, n_lower)
    y_lower = 0.1 + 0.8 * x_lower**2
    
    points = np.vstack([
        np.column_stack([x_upper, y_upper]),
        np.column_stack([x_lower, y_lower])
    ])
    
    # Small perturbation to ensure general position
    points += np.random.uniform(-0.003, 0.003, (n, 2))
    points = np.clip(points, 1e-4, 1 - 1e-4)
    
    return points


# EVOLVE-BLOCK-END
