import numpy as np

def heuristics(demands, capacity):
        n = len(demands)
        demands = np.array(demands)
        
        # 1. Construct Weight Matrix
        # Residual R_ij = C - (d_i + d_j)
        # W_ij = 1 / max(R_ij, eps) if R_ij >= 0 else 0
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
        # Shift to make non-negative
        coords = fiedler - np.min(fiedler)
        
        # Optional: Normalize coords to [0, 1] to keep heuristic values bounded
        max_coord = np.max(coords)
        if max_coord > 0:
            coords = coords / max_coord
        else:
            # Degenerate case
            coords = np.ones(n)
            
        # 5. Heuristic Matrix
        # H_ij = coords_i * coords_j
        H = np.outer(coords, coords)
        
        # Apply fitting mask: if not fitting, set to 1e-9
        H = np.where(fit_mask, H, 1e-9)
        
        # Ensure positive (should be, but just in case)
        H = np.where(H > 0, H, 1e-9)
        
        return H
