"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k.
    Actually, the multiplication tensor is: T[i,j,k] where the trilinear form
    is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] (with W convention).
    
    For matrix multiplication: (AB)[i,k] = sum_j A[i,j] B[j,k]
    So the tensor entry T[i,j,k] = 1 for all valid i,j,k (it's the identity
    trilinear form for multiplication).
    """
    T = np.ones((n, m, p))
    return T

def _check_decomposition(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = U.shape[1]
    T_reconstructed = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    T_target = np.ones((n, m, p))
    return np.allclose(T_reconstructed, T_target, atol=1e-10)

def _check_decomposition_fast(U, V, W, n, m, p):
    """Faster check using einsum."""
    rank = U.shape[1]
    u = U.reshape(n, m, rank)
    v = V.reshape(m, p, rank)
    w = W.reshape(p, n, rank)
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', u, v, w)
    T_target = np.ones((n, m, p))
    return np.max(np.abs(T_reconstructed - T_target)) < 1e-10

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank using iterative optimization
    # The multiplication tensor has some symmetry that can be exploited
    # Known: rank(M_{4,4,4}) <= 15 (Strassen-like), but for 4x4x9 we need to search
    
    # Use a simple approach: try ranks from lower bounds upward
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # Also: rank >= n*p = 36 (from the output dimension)
    
    # Actually, a better lower bound for matrix multiplication rank:
    # rank >= n*m, rank >= m*p, rank >= n*p
    # So rank >= 36 for our case
    
    # Let's try to find a decomposition using a greedy/iterative approach
    # For each rank r, try to solve: find U,V,W such that
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # Use alternating least squares style approach
    for target_rank in range(36, min(best_rank, 60)):
        if time.time() > deadline:
            break
        for attempt in range(5):
            if time.time() > deadline:
                break
            # Initialize randomly
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Iterative refinement: alternate fixing two factors and solving for the third
            # This is like ALS for tensor decomposition
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Fix V, W, solve for U:
                # For each (i,j): sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r] for all k) = 1
                # Actually: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Fix V,W: for each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is: for fixed (i,j), U[i*m+j,:] satisfies:
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # Let M[i,j] be a p x target_rank matrix with M[i,j,k,r] = V[j*p+k,r]*W[k*n+i,r]
                # Then M[i,j] @ U[i*m+j,:] = 1 (vector of ones of length p)
                
                # Similarly fix U,W solve for V, fix U,V solve for W
                
                # Solve for U given V,W
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            A_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A_mat @ U[i*m+j,:] = ones(p)
                        try:
                            sol, residuals, rank_, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            U[i * m + j, :] = sol
                        except:
                            pass
                
                # Solve for V given U,W
                for j in range(m):
                    for k in range(p):
                        # For each i: sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = 1
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            A_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            sol, residuals, rank_, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            V[j * p + k, :] = sol
                        except:
                            pass
                
                # Solve for W given U,V
                for k in range(p):
                    for i in range(n):
                        # For each j: sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            sol, residuals, rank_, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W[k * n + i, :] = sol
                        except:
                            pass
                
                # Check convergence
                if _check_decomposition_fast(U, V, W, n, m, p):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if target_rank < best_rank and _check_decomposition_fast(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
