import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    # Fallback scores: local heuristic
    degrees = adj.sum(axis=1).astype(float)
    nbr_deg_sums = adj @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)
    fallback_scores = -degrees + avg_nbr_deg

    try:
        if time.time() > deadline - 5:
            return fallback_scores
            
        # Sparse adjacency for eigsh
        A_sp = csr_matrix(adj.astype(float))
        
        # Get principal eigenvector (largest eigenvalue)
        # For undirected graphs, this gives a 2-coloring-like partition
        # Vertices with same sign tend to be non-adjacent (independent set candidates)
        k = 1
        v0 = np.ones(n) / np.sqrt(n)
        
        try:
            eigenvalues, eigenvectors = eigsh(A_sp, k=k, v0=v0, maxiter=1000, tol=1e-6)
            principal_vec = eigenvectors[:, 0]
        except Exception:
            # Fallback: use power iteration
            vec = np.ones(n) / np.sqrt(n)
            for _ in range(50):
                new_vec = A_sp @ vec
                norm = np.linalg.norm(new_vec)
                if norm < 1e-10:
                    break
                vec = new_vec / norm
            principal_vec = vec
        
        # The sign of the principal eigenvector gives a natural 2-coloring
        # Positive vertices form one part, negative form another
        # Each part is approximately an independent set
        
        # Score strategy:
        # 1. Primary: prefer vertices with larger |eigenvector component| (more "central" in their color class)
        # 2. Secondary: within same color, lower degree is better (less blocking)
        # 3. Tertiary: tie-break by vertex index for determinism
        
        abs_vec = np.abs(principal_vec)
        sign_vec = np.sign(principal_vec)
        
        # Normalize abs_vec to [0, 1]
        max_abs = np.max(abs_vec)
        if max_abs > 1e-10:
            normalized_abs = abs_vec / max_abs
        else:
            normalized_abs = np.zeros(n)
        
        # Combine: high |eigenvector| magnitude + low degree penalty
        # The eigenvector magnitude indicates how strongly the vertex belongs to its color class
        # Low degree means fewer neighbors to block
        
        # Weight: 0.7 for spectral magnitude, 0.3 for degree penalty
        degree_norm = degrees / (np.max(degrees) + 1e-10)
        scores = 0.7 * normalized_abs - 0.3 * degree_norm
        
        # Add small perturbation based on sign to break ties between color classes
        # This ensures we don't accidentally pick adjacent vertices from the same color class
        # if their scores are too close
        scores += sign_vec * 0.001
        
        return scores
        
    except Exception:
        return fallback_scores
