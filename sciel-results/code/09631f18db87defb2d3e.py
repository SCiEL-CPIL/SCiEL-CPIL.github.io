import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    coords = np.array(coordinates)
    depot = coords[0]
    
    # Vector from each node i to depot
    v_depot = depot - coords  # shape (n, 2)
    norm_depot = np.linalg.norm(v_depot, axis=1)  # shape (n,)
    
    # Vector from each node i to each node j
    # diff[i, j] = coords[j] - coords[i]
    diff = coords[None, :, :] - coords[:, None, :]  # shape (n, n, 2)
    norm_j = np.linalg.norm(diff, axis=2)  # shape (n, n)
    
    # Dot product between v_depot[i] and diff[i, j]
    # v_depot[i] is (2,), diff[i, j] is (2,)
    # We want dot[i, j] = sum(v_depot[i] * diff[i, j])
    # v_depot[:, None, :] shape (n, 1, 2)
    # diff shape (n, n, 2)
    dot = np.sum(v_depot[:, None, :] * diff, axis=2)  # shape (n, n)
    
    # Cosine of angle
    # Handle division by zero
    denom = norm_depot[:, None] * norm_j
    cos_theta = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-9)
    
    # Clip for arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angle
    angle = np.arccos(cos_theta)
    
    # Heuristic
    # eta[i, j] = 1 / (d[i, j] * (1 + angle[i, j]))
    # Avoid division by zero if d[i, j] == 0 (diagonal)
    d = distance_matrix
    eta = np.divide(1.0, d * (1.0 + angle), out=np.full_like(d, 1e-9), where=d > 1e-9)
    
    # Ensure positivity
    eta[eta <= 0] = 1e-9
    
    return eta
