import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    demands = np.array(demands)
    
    # 1. Construct Weight Matrix
    # Residual R_ij = C - (d_i + d_j)
    # We want to weight tight fits higher.
    sums = demands[:, None] + demands[None, :]
    residuals = capacity - sums
    
    # Mask for fitting pairs
    fit_mask = residuals >= 0
    
    # Initialize W
    W = np.zeros((n, n))
    
    # Calculate weights for fitting pairs
    # Avoid division by zero
    eps = 1e-6
    safe_residuals = np.where(fit_mask, residuals + eps, 1.0)
    # Inverse residual: smaller residual -> larger weight
    W = np.where(fit_mask, 1.0 / safe_residuals, 0.0)
    
    # Symmetric, diagonal is 0 (no self loops)
    np.fill_diagonal(W, 0)
    
    # 2. Compute Laplacian
    # L = D - W
    degrees = W.sum(axis=1)
    D = np.diag(degrees)
    L = D - W
    
    # 3. Eigendecomposition
    # eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Fiedler vector is eigenvector corresponding to 2nd smallest eigenvalue
    # Index 1
    fiedler = eigenvectors[:, 1]
    
    # 4. Compatibility Coordinates
    # Use the Fiedler vector as 1D coordinates.
    # We don't necessarily need to shift or normalize, as we will use distances.
    # However, normalizing helps with scaling for the exponential.
    # Let's normalize to [0, 1] to have a consistent scale for alpha.
    min_f = np.min(fiedler)
    max_f = np.max(fiedler)
    if max_f > min_f:
        coords = (fiedler - min_f) / (max_f - min_f)
    else:
        coords = np.zeros(n)
        
    # 5. Heuristic Matrix
    # Distance in spectral space
    dists = np.abs(coords[:, None] - coords[None, :])
    
    # Heuristic is decaying exponential of distance
    # Alpha controls the "sharpness". 
    # If alpha is too high, only very close items are attractive.
    # If alpha is too low, it's nearly uniform.
    alpha = 5.0
    H = np.exp(-alpha * dists)
    
    # Apply fitting mask: if not fitting, set to 1e-9
    H = np.where(fit_mask, H, 1e-9)
    
    # Ensure positive (should be, but just in case)
    H = np.where(H > 0, H, 1e-9)
    
    return H
