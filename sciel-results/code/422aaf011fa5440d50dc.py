import time
import sys
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s wall-clock limit, leave 10s margin
    
    # Target tensor T of shape (n*m, m*p, p*n) flattened?
    # Or just define the function to compute the residual.
    
    def flatten_params(x):
        """Unpack flat array into U, V, W."""
        rank = len(x) // (n*m + m*p + p*n)
        # This logic is wrong, x is flat of size rank * (n*m + m*p + p*n)
        # But we don't know rank inside.
        pass

    def residual(x, rank):
        """
        x: flat array of length rank * (n*m + m*p + p*n)
        Returns residual vector of length n*m*p
        """
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = size_u + size_v + size_w
        
        # Reshape x into U, V, W
        # x is shape (rank * total_size,)
        # We want U (size_u, rank), V (size_v, rank), W (size_w, rank)
        
        # Reshape to (rank, total_size) then split
        X = x.reshape(rank, total_size)
        U = X[:, :size_u].T  # (size_u, rank)
        V = X[:, size_u:size_u+size_v].T  # (size_v, rank)
        W = X[:, size_u+size_v:].T  # (size_w, rank)
        
        # Compute sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to match this to delta_{i,i'} delta_{j,j'} delta_{k,k'}
        # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 if indices match?
        # No, the multiplication tensor is defined such that
        # C[k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # And we want C[k*n+i] = sum_j A[i*m+j] B[j*p+k]
        # So the tensor T is such that T[i*m+j, j*p+k, k*n+i] = 1.
        
        # Let's build the target vector
        # Target vector y of length n*m*p?
        # No, the residual is over all (i,j,k) triples.
        # There are n*m*p triples.
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Target is 1.0
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact and entries are half-integers."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1.0) > 1e-8:
                        return False
        # Check half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    # Check if val is close to k/2
                    if abs(val * 2 - round(val * 2)) > 1e-5:
                        return False
        return True

    # Try ranks from low to high
    best_result = None
    best_rank = n * m * p # 36
    
    # Generate schoolbook decomposition as fallback
    U_school = [[0.0] * best_rank for _ in range(n * m)]
    V_school = [[0.0] * best_rank for _ in range(m * p)]
    W_school = [[0.0] * best_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1.0
                V_school[j * p + k][col] = 1.0
                W_school[k * n + i][col] = 1.0
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Search for lower ranks
    # Possible ranks: 18, 24, 27, 30, 33
    candidate_ranks = [18, 24, 27, 30, 33]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
                
            # Initialize with small random values
            size_u = n * m
            size_v = m * p
            size_w = p * n
            total_size = size_u + size_v + size_w
            
            x0 = np.random.randn(rank * total_size) * 0.1
            
            # Use least squares
            try:
                result = least_squares(residual, x0, args=(rank,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if result.cost < 1e-10:
                # Check if solution is valid (half-integers)
                X = result.x.reshape(rank, total_size)
                U = X[:, :size_u].T
                V = X[:, size_u:size_u+size_v].T
                W = X[:, size_u+size_v:].T
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                # Verify rounded version
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if verify_decomposition(U_list, V_list, W_list, n, m, p):
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
