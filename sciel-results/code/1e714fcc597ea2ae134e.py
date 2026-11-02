"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random
import numpy as np
from scipy.linalg import svd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Schoolbook gives rank 80. We attempt to find a lower-rank decomposition.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using iterative refinement
    # Strategy: Start from a random low-rank initialization and refine
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        # Reconstruct: C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For each (i,k), the coefficient of A[i,j]*B[j,k] must be 1
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    coeff = 0.0
                    for c in range(rank):
                        coeff += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(coeff - 1.0) > 1e-9:
                        return False
        # Check no spurious terms: for (i,k) where we don't expect contribution from A[i,j]*B[j,k]
        # Actually in matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k], so every (i,j,k) triple contributes exactly once
        # The decomposition should only produce the correct products
        return True
    
    # Strategy: Use the known structure. The tensor for matmul (n,m,p) has rank at most n*m*p.
    # Known bounds: For 4x4x5, the minimum rank is not known to be below 80 with simple methods,
    # but we can try randomized low-rank approximations and refine.
    
    # Alternative: Use the fact that we can try to merge some of the 80 terms.
    # If two terms (j1,k1) and (j2,k2) can be combined, we might reduce rank.
    
    # Let's try a direct approach: start with rank r and solve for U, V, W
    # using iterative scaling / multiplicative updates similar to CP decomposition.
    
    def try_rank(r, max_iters=200, tol=1e-12):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # The tensor T[i,j,k] = 1 if this corresponds to a matmul term, else 0
        # Specifically, T[(i,j), (j,k), (k,i)] = 1 for all i,j,k
        # In our indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The product sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] should equal 1
        
        # Use alternating minimization:
        # Fix V, W -> solve for U (linear)
        # Fix U, W -> solve for V (linear)
        # Fix U, V -> solve for W (linear)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Solve for U: For each (i,j), we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all k
            # This is: for each (i,j), the vector of length p: (U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:]) for k) = 1
            # Let's define for each (i,j): M_ij is a p x r matrix where M_ij[k,c] = V[j*p+k,c] * W[k*n+i,c]
            # Then we need U[i*m+j,:] @ M_ij = 1 (vector of ones)
            # This is r unknowns, p equations. If p < r, underdetermined; if p >= r, we can use least squares
            
            for i in range(n):
                for j in range(m):
                    # Build M_ij: p x r
                    M_ij = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            M_ij[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    # Solve U[i*m+j,:] @ M_ij = 1
                    # i.e., M_ij.T @ U[i*m+j,:] = 1
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M_ij.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V: For each (j,k), we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i
            # For each (j,k): M_jk is an n x r matrix where M_jk[i,c] = U[i*m+j,c] * W[k*n+i,c]
            # We need V[j*p+k,:] @ M_jk = 1
            for j in range(m):
                for k in range(p):
                    M_jk = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            M_jk[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M_jk.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W: For each (k,i), we need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all j
            # For each (k,i): M_ki is an m x r matrix where M_ki[j,c] = U[i*m+j,c] * V[j*p+k,c]
            # We need W[k*n+i,:] @ M_ki = 1
            for k in range(p):
                for i in range(n):
                    M_ki = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            M_ki[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M_ki.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        coeff = 0.0
                        for c in range(r):
                            coeff += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err += (coeff - 1.0) ** 2
            if err < tol:
                return U, V, W
        
        return None
    
    # Try ranks from 80 down
    for r in range(80, 55, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=100)
        if result is not None:
            U, V, W = result
            # Convert to list format
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
            # Verify
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
