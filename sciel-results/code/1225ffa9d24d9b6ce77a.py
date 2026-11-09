"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if U, V, W exactly decompose the matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    # U: (nm, rank), V: (mp, rank), W: (pn, rank)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] == 1 for all i,j,k
    
    rank = U.shape[1]
    # Efficient check: 
    # T is a 3D tensor. We can compute it via einsum or loops.
    # Given small sizes, loops are fine.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _als_step(U, V, W, n, m, p, rank):
    """One step of Alternating Least Squares."""
    nm, mp, pn = n * m, m * p, p * n
    
    # Fix V, W. Solve for U.
    # For each (i, j), we need sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])_k = 1 for all k.
    # This is a linear system for each (i,j) if we consider the vector of k's?
    # No, the equation is:
    # For fixed i, j:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k in 0..p-1.
    # Let A_{k,c} = V[j*p+k, c] * W[k*n+i, c].
    # We need sum_c U_{c} A_{k,c} = 1.
    # This is U^T A^T = 1_vector? 
    # A is (p, rank). U is (rank,).
    # We want U such that A^T U = 1_p.
    # If rank > p, we have underdetermined system. We want min norm or just any solution.
    # Actually, we want to minimize error over all i,j,k.
    # ALS usually minimizes the Frobenius norm of the error tensor.
    
    # Let's construct the linear system for U.
    # Variables: U (nm * rank)
    # Equations: for each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a linear system in U.
    # Matrix M_U of size (n*m*p, nm*rank).
    # M_U[(i*m+j)*p+k, (i*m+j)*rank + c] = V[j*p+k, c] * W[k*n+i, c]
    # This matrix is sparse.
    
    # For simplicity and speed, we can use a simplified update rule or just random projection.
    # Given the small size, let's build the matrix.
    num_eq = n * m * p
    num_var = nm * rank
    
    M = np.zeros((num_eq, num_var))
    b = np.ones(num_eq)
    
    for i in range(n):
        for j in range(m):
            u_row = i * m + j
            for k in range(p):
                eq_idx = (u_row * p) + k
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    M[eq_idx, u_row * rank + c] = V[v_idx, c] * W[w_idx, c]
                    
    # Solve M U_flat = b using least squares
    U_flat, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    U = U_flat.reshape(nm, rank)
    
    # Similarly for V
    M = np.zeros((num_eq, mp * rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = (i * m + j) * p + k
                u_idx = i * m + j
                v_row = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    M[eq_idx, v_row * rank + c] = U[u_idx, c] * W[w_idx, c]
    V_flat, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    V = V_flat.reshape(mp, rank)
    
    # Similarly for W
    M = np.zeros((num_eq, pn * rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = (i * m + j) * p + k
                u_idx = i * m + j
                v_idx = j * p + k
                w_row = k * n + i
                for c in range(rank):
                    M[eq_idx, w_row * rank + c] = U[u_idx, c] * V[v_idx, c]
    W_flat, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    W = W_flat.reshape(pn, rank)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Try to find a rank lower than schoolbook
    # Start from schoolbook_rank - 1 down to a reasonable limit
    # Limit search time to 60 seconds to be safe
    
    start_time = time.time()
    max_time = 60
    
    for rank in range(schoolbook_rank - 1, max(20, schoolbook_rank - 20), -1):
        if time.time() - start_time > max_time:
            break
            
        # Try a few random initializations
        for attempt in range(20):
            if time.time() - start_time > max_time:
                break
                
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Run ALS for a few iterations
            for _ in range(10):
                U, V, W = _als_step(U, V, W, n, m, p, rank)
                
                # Check for convergence or exactness after rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Also check if the unrounded version is close to integer
                if np.allclose(U, U_int, atol=1e-6) and np.allclose(V, V_int, atol=1e-6) and np.allclose(W, W_int, atol=1e-6):
                     if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}

    # Fall back to schoolbook
    rank = schoolbook_rank
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
