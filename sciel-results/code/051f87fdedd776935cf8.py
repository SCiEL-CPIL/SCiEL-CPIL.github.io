import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    n = distance_matrix.shape[0]
    
    # Base heuristic: Inverse distance with beta=3.0
    # Beta=4 was better than 5, so try 3.
    beta = 3.0
    eps = 1e-9
    heuristic = 1.0 / np.power(distance_matrix + eps, beta)
    
    # Angular Consistency Factor
    # Calculate angles relative to depot (0,0)
    # coordinates[0] is depot
    depot_x, depot_y = coordinates[0]
    x = coordinates[:, 0] - depot_x
    y = coordinates[:, 1] - depot_y
    
    angles = np.arctan2(y, x)
    
    # Pre-compute angular differences for all pairs
    # angle_diff[i, j] = angles[j] - angles[i]
    angle_diff = angles[None, :] - angles[:, None]
    
    # Normalize to [-pi, pi]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    # Factor: 
    # If |angle_diff| is small, it's a good move (local cluster)
    # If |angle_diff| is large, it's a jump across the map (bad)
    
    # Let's use a smooth function or step function.
    # Step function:
    # < pi/4 (45 deg): 1.2
    # > 3*pi/4 (135 deg): 0.8
    # Else: 1.0
    
    factor = np.ones_like(heuristic)
    
    # Mask for small angle diff
    mask_small = np.abs(angle_diff) < (np.pi / 4)
    factor[mask_small] = 1.2
    
    # Mask for large angle diff
    mask_large = np.abs(angle_diff) > (3 * np.pi / 4)
    factor[mask_large] = 0.8
    
    heuristic *= factor
    
    # Ensure positive
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
