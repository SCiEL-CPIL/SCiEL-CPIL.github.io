import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # Piecewise linear basis (hat functions) with 10 fixed knots
    # Knots evenly spaced across [0.1, 6]
    # We'll use 10 hat functions centered at 10 evenly spaced points
    # Each hat function is 0 outside [center - width, center + width]
    # and linearly rises to 1 at center, falls to 0 at edges
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Define 10 knot centers evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Width: distance between adjacent centers
    # For edge knots, use half-width to domain boundary
    width = (6.0 - 0.1) / 9.0  # spacing between centers
    
    basis = np.zeros((n_points, 10))
    
    for i in range(10):
        c = centers[i]
        left = c - width
        right = c + width
        
        # Hat function: 0 outside [left, right], peaks at 1 at c
        # Linear rise from left to c, linear fall from c to right
        mask_left = (x >= left) & (x < c)
        mask_right = (x > c) & (x <= right)
        mask_center = (x == c)
        
        basis[mask_left, i] = (x[mask_left] - left) / width
        basis[mask_right, i] = (right - x[mask_right]) / width
        basis[mask_center, i] = 1.0
    
    return basis @ params
