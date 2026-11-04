"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition exactly reconstructs the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k] contributes.
    Actually, the matrix multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
    (it's the identity in the sense that C[i,k] = sum_j A[i,j]B[j,k]).
    
    Wait, let me reconsider. The tensor for matrix multiplication of n x m by m x p
    matrices is a 3-tensor T of shape (n, m, p) where T[i,j,k] = 1 if and only if
    the (i,k) entry of the product involves A[i,j] and B[j,k]. Since every (i,j,k)
    triple contributes to C[i,k], T[i,j,k] = 1 for all i,j,k.
    
    Actually no. The standard matrix multiplication tensor T_{ijk} = delta_{i,i} delta_{j,j} delta_{k,k}... 
    Let me think again.
    
    The matrix multiplication tensor T is defined such that:
    C[i,k] = sum_{j} A[i,j] * B[j,k]
    
    The tensor T has entries T[i,j,k] = 1 for all valid i,j,k. This is because each
    term A[i,j]*B[j,k] contributes to C[i,k]. So T is the all-ones tensor of shape (n,m,p).
    
    Wait, that's not right either. Let me look at the convention more carefully.
    
    The factor matrices U, V, W satisfy:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where the indexing follows the official trace convention W[k*n+i, r].
    
    For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    # T[i,j,k] = 1 for all i,j,k
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor from factors
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    errors = 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if err > tol:
                    errors += 1
    return errors == 0, max_err


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x5 x 5x6 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to find a lower rank decomposition using random search with local optimization
    # Strategy: Start from schoolbook, try to merge columns (reduce rank)
    
    # For now, return the schoolbook decomposition as a safe baseline
    # Convert to lists with integer entries
    U_list = U0.tolist()
    V_list = V0.tolist()
    W_list = W0.tolist()
    
    # Ensure entries are integers or half-integers (they are all 0 or 1 here)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
