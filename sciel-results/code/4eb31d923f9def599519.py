import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if i<n, j<m, k<p else 0
    # Actually, the tensor for matrix multiplication is defined by:
    # T[i, j, k] = 1 if the indices correspond to a valid multiplication term?
    # No, the tensor is the structure constant.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # T[i, j, k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # So T[i, j, k] = 1 for all i, j, k in range.
    # Wait, the tensor is M_{n,m,p}.
    # The entries are 1 if (i,j,k) is a valid triple.
    # Yes, all 1s for i in 0..n-1, j in 0..m-1, k in 0..p-1.
    
    t_target = np.ones((n, m, p))
    t_flat = t_target.flatten()
    dim = n * m * p
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, dim))
    V_sb = np.zeros((m * p, dim))
    W_sb = np.zeros((p * n, dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = dim
    
    # Deadline guard
    start_time = time.time()
    deadline = 100.0 # 100s out of 120s
    
    # Search for lower rank
    # We try to find R < dim such that t_flat is in span of R random rank-1 tensors
    # We iterate R downwards
    for R in range(dim - 1, 50, -1): # Try down to 50
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this R
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Generate random rank-1 tensors
            # Use small integers to increase chance of integer coefficients
            # u, v, w vectors
            u = np.random.randint(-2, 3, size=n * m)
            v = np.random.randint(-2, 3, size=m * p)
            w = np.random.randint(-2, 3, size=p * n)
            
            # This generates 1 tensor. We need R tensors.
            # Actually, generating R random tensors and checking span is O(R^2 * dim)
            # dim=60, R=50. 50^2 * 60 = 150,000 ops. Very fast.
            
            tensors = []
            for _ in range(R):
                u = np.random.randint(-2, 3, size=n * m)
                v = np.random.randint(-2, 3, size=m * p)
                w = np.random.randint(-2, 3, size=p * n)
                # Outer product
                t_i = np.einsum('i,j,k->ijk', u, v, w).flatten()
                tensors.append(t_i)
                
            M = np.column_stack(tensors)
            
            # Check if t_flat is in span of M
            # Solve M c = t_flat
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(M, t_flat, rcond=None)
                # Check residual
                if residuals and residuals[0] < 1e-9:
                    # Check if coeffs are close to integers
                    int_coeffs = np.round(coeffs)
                    if np.allclose(M @ int_coeffs, t_flat, atol=1e-9):
                        # Found a solution!
                        # Construct U, V, W
                        U = np.zeros((n * m, R))
                        V = np.zeros((m * p, R))
                        W = np.zeros((p * n, R))
                        # Restart with storage
                        pass
            except:
                pass
        
        # Optimization: Store the vectors
        # I'll refactor the search to store vectors
        
    # If no better found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
