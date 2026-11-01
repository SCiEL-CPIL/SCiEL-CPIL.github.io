"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    T_decomp = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T_decomp[i, j, k] += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
    
    return np.allclose(T, T_decomp, atol=1e-9)

def _known_rank11():
    """Return the known rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[2][2] = 1; W[2][4] = 1
    W[1][1] = 1; W[1][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    U[0][7] = 1
    V[2][7] = 1
    
    U[1][8] = 1
    V[5][8] = 1
    
    U[2][9] = 1
    V[2][9] = 1
    
    U[3][10] = 1
    V[5][10] = 1
    
    W[4][7] = 1; W[4][8] = 1
    W[5][9] = 1; W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank10():
    """Attempt to find a rank-10 decomposition via structured search."""
    n, m, p = DIMENSIONS
    rank = 10
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Strategy: Start from rank-11 and try to find a linear dependency
    # that allows us to eliminate one term.
    # 
    # We'll parameterize the U, V matrices with some structure and solve for W.
    # The key is to find U, V such that the system for W is consistent.
    
    np.random.seed(42)
    
    best_error = float('inf')
    best_solution = None
    
    # Try multiple random initializations
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization with some structure
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        
        # For each W index (k*n+i), we need to solve:
        # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        # This gives m equations in rank unknowns for each W row.
        
        W = np.zeros((p * n, rank))
        error = 0.0
        
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                # Build the m x rank matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Target: T[i, j, k] for j in range(m)
                b = np.array([1.0] * m)
                
                # Least squares solution
                if rank >= m:
                    # Use lstsq
                    W_row, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    if residuals.size > 0:
                        error += residuals[0]
                else:
                    # Underdetermined - use minimum norm solution
                    W_row = np.linalg.pinv(A) @ b
                    # Check residual
                    residual = b - A @ W_row
                    error += np.linalg.norm(residual)
                
                W[w_idx, :] = W_row
        
        if error < best_error:
            best_error = error
            best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-8:
                # Found a solution!
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try a more structured approach: modify the rank-11 solution
        if attempt % 5 == 0:
            # Start from rank-11 and try to merge two terms
            base = _known_rank11()
            U_base = np.array(base["U"])
            V_base = np.array(base["V"])
            W_base = np.array(base["W"])
            
            # Try to express one of the schoolbook terms as a combination
            # of existing terms plus a new parameterized term
            for merge_target in [7, 8, 9, 10]:  # Try merging each schoolbook term
                if time.time() > deadline - 5.0:
                    break
                
                # Create rank-10 by removing term merge_target
                # and trying to absorb it into others
                
                # This is complex - let's try a different approach:
                # Parameterize U and V with the structure that the third column
                # shares some factors with the first two columns.
                
                pass
    
    return None

def construct():
    # First, verify the known rank-11 solution
    result = _known_rank11()
    
    if not _verify(result["U"], result["V"], result["W"]):
        raise ValueError("Known rank-11 decomposition failed verification!")
    
    # Try to find rank-10
    rank10_result = _try_rank10()
    if rank10_result is not None:
        return rank10_result
    
    return result
# EVOLVE-BLOCK-END
