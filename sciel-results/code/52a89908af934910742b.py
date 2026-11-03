"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin before 140s hard limit
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions using random initialization + iterative refinement
    # For matrix multiplication (n x m) * (m x p) -> (n x p)
    # The tensor has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # Rank r decomposition: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Known bounds: For 3x4 * 4x7, schoolbook is 84.
    # We'll try to find better decompositions using optimization.
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        if U_mat.shape[1] != V_mat.shape[1] or U_mat.shape[1] != W_mat.shape[1]:
            return False
        if U_mat.shape[0] != n * m or V_mat.shape[0] != m * p or W_mat.shape[0] != p * n:
            return False
        
        # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
        # where delta is 1 if this is a valid triple, 0 otherwise
        # Actually, the matrix multiplication tensor is:
        # T[i,j,k] = 1 if the triple (i,j,k) corresponds to multiplying row i of A by col k of B
        # Wait, let me reconsider. The standard matrix multiplication tensor for (n x m) * (m x p):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 (it's the identity in the j-index sense)
        # More precisely, T[i,j,k] = 1 for all valid i,j,k, and the multiplication is:
        # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Let's verify by checking that the decomposition reproduces T[i,j,k] = 1 for all i,j,k
        # Actually no. Let me re-read the problem.
        
        # The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix:
        # The input is A (n x m) and B (m x p), output C (n x p) where C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # In tensor decomposition form: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
        # 
        # Wait, that's not quite right either. Let me think more carefully.
        #
        # The standard tensor decomposition of matrix multiplication:
        # We want: C[i,k] = sum_j A[i,j] B[j,k]
        # This can be written as: C[i,k] = sum_c (sum_j U_{i,j,c} A[i,j]) * (sum_j V_{j,k,c} B[j,k]) * W_{i,k,c}
        # where U, V, W are the factor matrices.
        #
        # For this to work for all A and B, we need:
        # sum_c W_{i,k,c} * (sum_j U_{i,j,c} A[i,j]) * (sum_j V_{j,k,c} B[j,k]) = sum_j A[i,j] B[j,k]
        #
        # This is equivalent to saying that the tensor T[i,j,k] = 1 (for all i,j,k) can be decomposed as:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        #
        # So we need: for all i in [0,n), j in [0,m), k in [0,p):
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try various ranks starting from a lower bound
    # Lower bound: at least max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    # Actually, the rank must be at least the matrix multiplication rank lower bound
    # For 3x4*4x7, known best is likely around 30-40
    
    # Let's try random initializations with optimization for ranks below schoolbook
    min_rank_to_try = 25
    max_rank_to_try = best_rank - 1
    
    np.random.seed(42)
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization with small values
            U_rand = np.random.randn(n * m, rank) * 0.1
            V_rand = np.random.randn(m * p, rank) * 0.1
            W_rand = np.random.randn(p * n, rank) * 0.1
            
            # Use simple iterative refinement: alternate between fixing two factors and solving for the third
            # This is a basic ALS (Alternating Least Squares) approach
            
            for iteration in range(200):
                if time.time() > deadline - 5.0:
                    break
                
                # Fix V and W, solve for U
                # For each (i*m+j), we need: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # This is a system of p equations for rank unknowns per row of U
                U_new = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        # This is: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        # Let M_k,c = V[j*p+k,c] * W[k*n+i,c], then sum_c U[i*m+j,c] * M_k,c = 1
                        # So U[i*m+j,:] @ M[:,k] = 1 for all k, i.e., M @ U[i*m+j,:] = ones(p)
                        # Where M is p x rank matrix with M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        M = np.zeros((p, rank))
                        for k in range(p):
                            for c in range(rank):
                                M[k, c] = V_rand[j*p+k, c] * W_rand[k*n+i, c]
                        # Solve M @ u = ones(p) for u (rank-vector)
                        # Use least squares
                        if rank >= p:
                            u, residuals, rank_sys, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        else:
                            # Underdetermined, use pseudoinverse
                            u = np.linalg.pinv(M) @ np.ones(p)
                        U_new[i*m+j, :] = u
                
                V_new = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        # For each j,p+k, need: sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
                        M = np.zeros((n, rank))
                        for i in range(n):
                            for c in range(rank):
                                M[i, c] = U_new[i*m+j, c] * W_rand[k*n+i, c]
                        if rank >= n:
                            v, residuals, rank_sys, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        else:
                            v = np.linalg.pinv(M) @ np.ones(n)
                        V_new[j*p+k, :] = v
                
                W_new = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        # For each k*n+i, need: sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
                        M = np.zeros((m, rank))
                        for j in range(m):
                            for c in range(rank):
                                M[j, c] = U_new[i*m+j, c] * V_new[j*p+k, c]
                        if rank >= m:
                            w, residuals, rank_sys, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        else:
                            w = np.linalg.pinv(M) @ np.ones(m)
                        W_new[k*n+i, :] = w
                
                U_rand, V_rand, W_rand = U_new, V_new, W_new
            
            # Check if we found a valid decomposition
            if verify_decomposition(U_rand, V_rand, W_rand, tol=1e-8):
                # Round to nearest half-integer if close
                U_rounded = np.round(U_rand * 2) / 2
                V_rounded = np.round(V_rand * 2) / 2
                W_rounded = np.round(W_rand * 2) / 2
                
                if verify_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-6):
                    best_rank = rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
                else:
                    best_rank = rank
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    break
    
    # Convert to lists as required
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
