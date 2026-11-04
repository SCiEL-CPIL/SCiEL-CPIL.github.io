import numpy as np
import time
from scipy.linalg import eigh

def solve(a, b):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = a.shape[0]
    
    # 1. Initialization via MDS and Procrustes
    # Center the matrices
    a_centered = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    b_centered = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    
    # Eigen-decomposition
    # We want the largest eigenvalues for MDS
    eigvals_a, eigvecs_a = eigh(a_centered)
    eigvals_b, eigvecs_b = eigh(b_centered)
    
    # Take top k dimensions (k=3 is usually sufficient for QAP)
    k = 3
    # Sort eigenvalues in descending order
    idx_a = np.argsort(eigvals_a)[::-1]
    idx_b = np.argsort(eigvals_b)[::-1]
    
    X_a = eigvecs_a[:, idx_a[:k]] * np.sqrt(np.maximum(eigvals_a[idx_a[:k]], 0))
    X_b = eigvecs_b[:, idx_b[:k]] * np.sqrt(np.maximum(eigvals_b[idx_b[:k]], 0))
    
    # Procrustes alignment
    # Minimize ||X_a - X_b R||_F
    # R = (X_b^T X_a)^+ (X_b^T X_a)^+^T ... wait
    # Standard Procrustes: R = V U^T where X_b^T X_a = U S V^T
    H = X_b.T @ X_a
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    
    # Align X_b to X_a
    X_b_aligned = X_b @ R
    
    # Compute initial permutation
    # Match rows of X_a to rows of X_b_aligned
    # Using Hungarian algorithm on the distance matrix
    from scipy.optimize import linear_sum_assignment
    
    # Compute distance matrix between X_a and X_b_aligned
    # D[i, j] = ||X_a[i] - X_b_aligned[j]||^2
    # D = sum(X_a^2) + sum(X_b_aligned^2) - 2 * X_a @ X_b_aligned.T
    dist_matrix = np.sum(X_a**2, axis=1)[:, np.newaxis] + np.sum(X_b_aligned**2, axis=1)[np.newaxis, :] - 2 * (X_a @ X_b_aligned.T)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # 2. Local Search (2-opt)
    # Precompute row sums of A and B for potential speedup, though delta calculation is O(n)
    
    def compute_cost(perm):
        # Cost = sum a[i,j] * b[perm[i], perm[j]]
        # Vectorized: (A * B[perm][:, perm]).sum()
        return np.sum(a * b[np.ix_(perm, perm)])
    
    current_cost = compute_cost(p)
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Find best swap
        # We can iterate over all pairs (i, j)
        # To speed up, we can break early if time is low, but n=150 is small enough
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n):
                # Calculate delta for swapping i and j
                k1 = p[i]
                k2 = p[j]
                
                # Delta = sum_{m != i, j} (a[i,m] - a[j,m]) * (b[k1, p[m]] - b[k2, p[m]])
                # We can vectorize this sum
                # Create mask for m != i, j
                # Or just sum all and subtract i, j terms
                # Term for m: (a[i,m] - a[j,m]) * (b[k1, p[m]] - b[k2, p[m]])
                
                # Let's compute the full sum and subtract i, j contributions
                # Full sum S = sum_m (a[i,m] - a[j,m]) * (b[k1, p[m]] - b[k2, p[m]])
                
                # Vectorized computation
                diff_a = a[i] - a[j]
                diff_b = b[k1, p] - b[k2, p]
                s = np.dot(diff_a, diff_b)
                
                # Subtract contributions where m=i and m=j
                # m=i: (a[i,i] - a[j,i]) * (b[k1, p[i]] - b[k2, p[i]]) = (a[i,i] - a[j,i]) * (b[k1, k1] - b[k2, k1])
                # m=j: (a[i,j] - a[j,j]) * (b[k1, p[j]] - b[k2, p[j]]) = (a[i,j] - a[j,j]) * (b[k1, k2] - b[k2, k2])
                
                term_i = (a[i, i] - a[j, i]) * (b[k1, k1] - b[k2, k1])
                term_j = (a[i, j] - a[j, j]) * (b[k1, k2] - b[k2, k2])
                
                delta = s - term_i - term_j
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i != -1 and best_delta < 0:
            # Perform swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
    
    return p.tolist()
