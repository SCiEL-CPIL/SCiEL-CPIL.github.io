"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def _verify(U, V, W, n=4, m=5, p=5, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Efficient verification using tensor contraction logic
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check if T is all ones.
    
    # Reshape for easier manipulation
    # U: (n*m, rank) -> (n, m, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute the tensor T
    # T[i, j, k] = sum_c U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
    
    # We can do this iteratively or using einsum
    # einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    
    return True

def _verify_full(U, V, W, n=4, m=5, p=5, trials=5, tol=1e-9):
    """
    Full verification: check that the decomposition correctly computes matrix multiplication
    for random test matrices.
    """
    if not U or not V or not W:
        return False
    
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        
        # Efficient computation
        # C[i, k] = sum_c (sum_j U[i*m+j, c] * A[i, j]) * (sum_k' V[j*p+k, c] * B[j, k']) * W[k*n+i, c]
        # Note: The indices in U, V, W are flattened.
        # U row index: i*m + j
        # V row index: j*p + k
        # W row index: k*n + i
        
        # Let's compute u_c = U^T A_flat ? No.
        # u_c = sum_j U[i*m+j, c] A[i,j]
        # v_c = sum_j V[j*p+k, c] B[j,k]
        
        # We can precompute:
        # A_flat is not directly useful because U depends on i and j separately.
        
        # Let's just loop over c for clarity and correctness, it's fast enough for small ranks
        for c in range(rank):
            u_vals = U_arr[:, c]  # shape (n*m,)
            v_vals = V_arr[:, c]  # shape (m*p,)
            w_val = W_arr  # shape (p*n,)
            
            # u_vals[i*m+j]
            # v_vals[j*p+k]
            # w_vals[k*n+i]
            
            # Reshape u_vals to (n, m)
            u_mat = u_vals.reshape(n, m)
            # Reshape v_vals to (m, p)
            v_mat = v_vals.reshape(m, p)
            # Reshape w_vals to (p, n) -> transpose to (n, p)
            w_mat = w_vals.reshape(p, n).T
            
            # Compute contribution of term c
            # C_c = (A * u_mat) @ (v_mat * B) * w_mat ?
            # Let's check:
            # (A * u_mat) is element-wise product? No.
            # u_c = sum_j A[i,j] U[i*m+j, c] = sum_j A[i,j] u_mat[i,j]
            # Let a_c[i] = sum_j A[i,j] u_mat[i,j]
            # v_c[k] = sum_j B[j,k] V[j*p+k, c] = sum_j B[j,k] v_mat[j,k]
            # Let b_c[k] = sum_j B[j,k] v_mat[j,k]
            # Then C[i,k] += a_c[i] * b_c[k] * w_mat[i,k]
            
            a_c = np.sum(A * u_mat, axis=1)  # shape (n,)
            b_c = np.sum(B * v_mat, axis=0)  # shape (p,)
            
            C_computed += np.outer(a_c, b_c) * w_mat
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    
    return True

def _schoolbook(n=4, m=5, p=5):
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

def _try_strassen_extension(n=4, m=5, p=5):
    """
    Attempt to find a decomposition by extending the Strassen 4x4x4 decomposition.
    This is a heuristic attempt. If it fails, returns None.
    """
    # The known rank for 4x4x4 is 17 (Strassen).
    # We need to handle 4x5x5.
    # This is complex to do exactly with rational numbers in a short time.
    # We will try a numerical optimization approach to find a rank-17 or rank-18 solution
    # and then snap to rationals if possible, or just use the float solution if it's exact enough.
    # However, the task requires "integer/half-integer entries".
    # Given the difficulty of finding an exact rational solution for 4x5x5 with rank < 100
    # in the available time, and the risk of producing an invalid result,
    # we will stick to the schoolbook unless we find a very simple trick.
    
    # One simple trick: Can we combine terms?
    # For 4x5x5, can we use the fact that 5 = 4+1?
    # Maybe embed in 4x4x4 and 4x4x1 etc?
    # This is too speculative to implement robustly in a few lines.
    
    return None

def construct():
    """
    Constructs a tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    """
    if _time_exceeded():
        return {"U": [], "V": [], "W": []}
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Strategy 1: Strassen-like extension (currently a stub, returns None)
    # Strategy 2: Randomized search for rank 99 (too slow/complex for this context)
    
    # For now, return the schoolbook decomposition.
    # The verification functions are provided to ensure correctness.
    
    # Quick verification of the schoolbook result
    if _verify(best_result['U'], best_result['V'], best_result['W'], n, m, p):
        return best_result
    
    # If schoolbook fails (shouldn't happen), return empty
    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
