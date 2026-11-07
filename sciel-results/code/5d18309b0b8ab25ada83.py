import numpy as np
import time
import sys

def heuristics(demands, capacity):
    start_time = time.time()
    deadline = 320.0 # Safety margin
    
    n = len(demands)
    d = np.array(demands, dtype=np.float64)
    
    # Fallback heuristic
    def fallback_heuristic():
        sums = d[:, None] + d[None, :]
        valid = sums <= capacity
        # Simple local heuristic: prefer larger sums
        h = np.where(valid, sums / capacity, 1e-9)
        return h

    try:
        # 1. Build Weighted Adjacency Matrix
        # Weight by the sum of demands, which favors tight packing
        sums = d[:, None] + d[None, :]
        valid = sums <= capacity
        
        # Initialize A
        A = np.zeros((n, n), dtype=np.float64)
        A[valid] = sums[valid]
        
        # 2. Compute Laplacian L = D - A
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        
        # 3. Spectral Embedding
        # Check time
        if time.time() > deadline - 5:
            return fallback_heuristic()
            
        try:
            import scipy.linalg as la
            eigenvalues, eigenvectors = la.eigh(L)
        except Exception:
            return fallback_heuristic()
            
        # Select k smallest non-zero eigenvalues
        tol = 1e-8
        non_zero_idx = np.where(eigenvalues > tol)[0]
        
        k = 3
        if len(non_zero_idx) < k:
            k = len(non_zero_idx)
            
        if k == 0:
            return fallback_heuristic()
            
        selected_eigenvectors = eigenvectors[:, non_zero_idx[:k]]
        
        # 4. Compute Pairwise Distances in Embedded Space
        X = selected_eigenvectors
        norms_sq = np.sum(X**2, axis=1)
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * (X @ X.T)
        dist_sq = np.maximum(dist_sq, 0)
        
        # 5. Compute Similarity
        non_zero_dists = dist_sq[dist_sq > 1e-6]
        if len(non_zero_dists) > 0:
            median_dist = np.median(non_zero_dists)
            sigma_sq = median_dist / 2 if median_dist > 0 else 1.0
        else:
            sigma_sq = 1.0
            
        similarity = np.exp(-dist_sq / (2 * sigma_sq))
        
        # 6. Compute Local Fitness
        waste = capacity - sums
        sigma_w = capacity / 4.0
        local_fitness = np.exp(-(waste**2) / (2 * sigma_w**2))
        
        # Combine
        H = local_fitness * (1 + similarity)
        
        # Apply mask
        H = np.where(valid, H, 1e-9)
        
        # Ensure symmetry and positivity
        H = (H + H.T) / 2
        H = np.maximum(H, 1e-9)
        
        # Check for non-finite values
        if not np.all(np.isfinite(H)):
            return fallback_heuristic()
            
        return H
        
    except Exception as e:
        return fallback_heuristic()
