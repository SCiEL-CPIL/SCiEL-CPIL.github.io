"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other (i,j,k) combinations not corresponding to valid multiplications, should be 0
    # Actually, the tensor is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Build the target tensor
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # Each term A[i,j]*B[j,k] contributes to C[i,k]
    
    # Compute the reconstructed tensor from the decomposition
    # T_reconstructed[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_rec = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_rec[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    return np.allclose(T, T_rec, atol=1e-10)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(n, m, p, target_rank, time_budget=30):
    """Try to find a decomposition of rank <= target_rank using random initialization and refinement."""
    start_time = time.time()
    
    # The tensor we want to decompose: T[i,j,k] = 1 for all valid (i,j,k) in [0,n)x[0,m)x[0,p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    num_rows_U = n * m
    num_rows_V = m * p
    num_rows_W = p * n
    
    best_result = None
    
    for attempt in range(50):
        if time.time() - start_time > time_budget:
            break
            
        # Random initialization with small values
        U = np.random.randn(num_rows_U, target_rank) * 0.1
        V = np.random.randn(num_rows_V, target_rank) * 0.1
        W = np.random.randn(num_rows_W, target_rank) * 0.1
        
        # Iterative refinement: fix two factors, solve for the third
        for iteration in range(200):
            if time.time() - start_time > time_budget:
                break
                
            # Fix V and W, solve for U
            # For each row i*m+j of U: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            # This is a system: for fixed (i,j), we need sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
            # So: U[i*m+j, :] @ (V[j*p+0:p, :] * W[0:n, :].T) = [1, 1, ..., 1] (length p)
            # But this is underdetermined if target_rank < p
            
            # Alternative approach: use ALS (Alternating Least Squares)
            # Minimize ||T - sum_c u_c * v_c * w_c||_F^2
            
            # Fix V and W, update U
            for i in range(n):
                for j in range(m):
                    # Target: T[i,j,k] = 1 for all k in [0,p)
                    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each k
                    # Build matrix A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        for c in range(target_rank):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    b = np.ones(p)
                    # Solve least squares: A @ u = b
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j, :] = u_sol
                    except:
                        pass
            
            # Fix U and W, update V
            for j in range(m):
                for k in range(p):
                    # Target: T[i,j,k] = 1 for all i in [0,n)
                    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each i
                    # Build matrix A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        for c in range(target_rank):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    b = np.ones(n)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k, :] = v_sol
                    except:
                        pass
            
            # Fix U and V, update W
            for k in range(p):
                for i in range(n):
                    # Target: T[i,j,k] = 1 for all j in [0,m)
                    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for each j
                    # Build matrix A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    b = np.ones(m)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i, :] = w_sol
                    except:
                        pass
            
            # Check convergence
            T_rec = np.zeros((n, m, p), dtype=np.float64)
            for c in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_rec[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            error = np.max(np.abs(T_rec - 1.0))
            if error < 1e-8:
                # Found a valid decomposition
                U_list = [[float(U[i, c]) for c in range(target_rank)] for i in range(num_rows_U)]
                V_list = [[float(V[i, c]) for c in range(target_rank)] for i in range(num_rows_V)]
                W_list = [[float(W[i, c]) for c in range(target_rank)] for i in range(num_rows_W)]
                
                # Verify with exact arithmetic if possible
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
        
        # Try a different random seed
        np.random.seed(attempt * 12345 + 67890)
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Try to reduce rank
    # Known lower bounds: for m x n matrix multiplication, rank >= max(n*m, m*p, p*n) = 48 in this case
    # But there might be better decompositions. Let's try ranks from 40 down to 30.
    
    start_time = time.time()
    time_limit = 100  # Leave 20s margin for the 120s limit
    
    for target_rank in range(40, 29, -1):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_reduce_rank(n, m, p, target_rank, time_budget=15)
        if result is not None:
            best = result
            break
    
    return best
# EVOLVE-BLOCK-END

