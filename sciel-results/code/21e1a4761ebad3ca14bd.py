import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    # Ensure we work with floats
    d = distance_matrix.astype(float)
    
    # Heuristic: Inverse distance squared with softened denominator
    # The +1e-4 term caps the maximum heuristic value for very close edges,
    # reducing extreme bias that can trap ants in local optima
    heuristic = 1.0 / (d**2 + 1e-4)
    
    # Set diagonal to 0 (or very small) to ensure no self-loop attraction
    np.fill_diagonal(heuristic, 0.0)
    
    # Replace non-positive or non-finite with 1e-9 as per task contract
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
