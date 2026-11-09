"""Geometric heuristic: use Z-order curve structure to guide ACO for TSP in unit square."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    
    if n <= 1:
        return np.ones((n, n), dtype=float)
    
    # Extract coordinates from distance matrix
    # For points in unit square, we can recover approximate coordinates using
    # the fact that D[i,j] = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    # Use MDS (multidimensional scaling) to recover 2D coordinates
    from scipy.spatial.distance import squareform
    
    # Recover coordinates via classical MDS
    # D is the distance matrix
    # Compute double-centered Gram matrix
    D2 = D ** 2
    # Double centering: B = -0.5 * J * D2 * J where J = I - (1/n)*11'
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2 eigenvalues
    # Clip negative eigenvalues to 0
    eigvals = np.maximum(eigvals, 0)
    
    # Coordinates: X = V * sqrt(lambda)
    X = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    
    # Normalize to unit square
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    if x_max > x_min:
        X[:, 0] = (X[:, 0] - x_min) / (x_max - x_min)
    else:
        X[:, 0] = 0.5
    
    if y_max > y_min:
        X[:, 1] = (X[:, 1] - y_min) / (y_max - y_min)
    else:
        X[:, 1] = 0.5
    
    # Compute Z-order (Morton) code for each point
    # Use 8 bits per coordinate (256 levels)
    bits = 8
    def morton_code(x, y, bits):
        """Compute Morton code from 2D coordinates in [0,1]."""
        # Scale to integer grid
        xi = int(x * (2**bits - 1))
        yi = int(y * (2**bits - 1))
        
        # Interleave bits
        code = 0
        for i in range(bits):
            # Get i-th bit from xi and yi
            xi_bit = (xi >> i) & 1
            yi_bit = (yi >> i) & 1
            # Interleave: yi_bit goes to even positions, xi_bit to odd
            code |= (xi_bit << (2 * i)) | (yi_bit << (2 * i + 1))
        
        return code
    
    # Compute Morton codes for all points
    morton_codes = np.array([morton_code(X[i, 0], X[i, 1], bits) for i in range(n)])
    
    # Sort by Morton code to get Z-order sequence
    z_order = np.argsort(morton_codes)
    
    # Build heuristic: strongly favor edges that are consecutive in Z-order
    # Also use inverse distance as base
    diag = np.eye(n, dtype=bool)
    safe_D = D.copy()
    safe_D[diag] = 1.0
    inv_dist = 1.0 / safe_D
    
    # Create Z-order proximity matrix
    # H_z[i,j] = 1/|pos_i - pos_j| in Z-order (smaller = closer in Z-order)
    # Actually, use exponential decay: H_z[i,j] = exp(-alpha * |zpos_i - zpos_j|)
    zpos = np.arange(n, dtype=float)
    zpos_sorted = zpos[z_order]  # zpos in Z-order sequence
    
    # For each pair (i,j), compute their positions in Z-order
    # Create inverse mapping: z_rank[i] = position of i in Z-order
    z_rank = np.empty(n, dtype=int)
    for pos, node in enumerate(z_order):
        z_rank[node] = pos
    
    # Compute Z-order distance matrix
    z_dist = np.abs(z_rank[:, None] - z_rank[None, :])
    
    # Exponential decay with alpha
    alpha = 0.5
    H_z = np.exp(-alpha * z_dist)
    H_z[diag] = 0
    
    # Combine: H = inv_dist * (1 + beta * H_z)
    # This makes edges that are both short AND close in Z-order more attractive
    beta = 5.0
    H = inv_dist * (1.0 + beta * H_z)
    
    # Ensure positive and finite
    H[diag] = 1e-9
    H = np.maximum(H, 1e-9)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
