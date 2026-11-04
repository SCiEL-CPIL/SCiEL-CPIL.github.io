"""Find rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is a valid index
    # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
    # Contract: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    r = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D: U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    
    # For each (i,j,k), compute sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
    # Use einsum
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # All entries should be 1
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _try_random_rank(r, n=2, m=4, p=5, max_iters=2000, timeout=10):
    """Try to find a valid decomposition of rank r via random init + local search."""
    start = time.time()
    best_err = float('inf')
    
    for attempt in range(max_iters):
        if time.time() - start > timeout:
            break
        
        # Random initialization
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        # Local refinement: alternate between fixing two and optimizing the third
        # Using gradient-free approach: for each rank element, adjust to reduce error
        for it in range(50):
            if time.time() - start > timeout:
                break
            
            # Compute current error tensor
            result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = result - 1.0
            best_err = min(best_err, np.max(np.abs(err)))
            
            if best_err < 1e-10:
                # Found valid solution
                return U.tolist(), V.tolist(), W.tolist()
            
            # Gradient descent step on W (fastest to update)
            # dW[k,i,c] = sum_{j} V[j,k,c] * (U[i,j,c] * err[i,j,k])
            # Simplified: just do a small random perturbation with acceptance
            # Actually let's do proper alternating minimization
            
            # Optimize W given U, V:
            # For each (k,i,c): W[k,i,c] should satisfy sum_c W[k,i,c]*U[i,j,c]*V[j,k,c] = 1 for all j
            # This is a linear system in W for each (k,i)
            # For fixed (k,i), we need: sum_c W[k,i,c] * (sum_j U[i,j,c]*V[j,k,c] * delta) ... 
            # Actually the constraint is: for all i,j,k: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # For fixed (k,i), this gives us constraints over j: sum_c W[k,i,c]*U[i,j,c]*V[j,k,c] = 1
            # That's m equations in r unknowns (W[k,i,0..r-1])
            
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, sum_c W[k,i,c] * U[i,j,c] * V[j,k,c] = 1
                    # A[j,c] = U[i,j,c] * V[j,k,c], b[j] = 1
                    A_mat = np.zeros((m, r))
                    for j in range(m):
                        A_mat[j, :] = U3[i, j, :] * V3[j, k, :]
                    b_vec = np.ones(m)
                    
                    # Solve least squares
                    try:
                        W_new, residuals, rank_, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W3[k, i, :] = W_new
                    except:
                        pass
            
            # Optimize U given V, W:
            for i in range(n):
                for j in range(m):
                    # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all k
                    A_mat = np.zeros((p, r))
                    for k in range(p):
                        A_mat[k, :] = V3[j, k, :] * W3[k, i, :]
                    b_vec = np.ones(p)
                    try:
                        U_new, residuals, rank_, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U3[i, j, :] = U_new
                    except:
                        pass
            
            # Optimize V given U, W:
            for j in range(m):
                for k in range(p):
                    # sum_c V[j,k,c]*U[i,j,c]*W[k,i,c] = 1 for all i
                    A_mat = np.zeros((n, r))
                    for i in range(n):
                        A_mat[i, :] = U3[i, j, :] * W3[k, i, :]
                    b_vec = np.ones(n)
                    try:
                        V_new, residuals, rank_, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V3[j, k, :] = V_new
                    except:
                        pass
        
        # Check final result
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(result - 1.0))
        best_err = min(best_err, err)
        if err < 1e-10:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Try ranks from theoretical minimum upward
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    min_rank = max(n * m, m * p, p * n)
    max_rank = n * m * p  # Schoolbook
    
    # Try to find a valid decomposition at each rank
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random attempts at this rank
        timeout_per_rank = min(15, max(1, (deadline - 5 - time.time()) // 3))
        result = _try_random_rank(r, n, m, p, max_iters=100, timeout=timeout_per_rank)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition (always valid, rank = n*m*p)
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
