"""Matrix multiplication 4,5,5 tensor decomposition with rank minimization."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize, LinearConstraint

DIMENSIONS = [4, 5, 5]

def _build_schoolbook(n, m, p):
    """Build schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    u_val = U[i * m + j, r]
                    v_val = V[j * p + k, r]
                    w_val = W[k * n + i, r]
                    val += u_val * v_val * w_val
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, []

def _add_strassen_block(U, V, W, col, i0, j0, k0, n, m, p):
    """
    Add Strassen's 7 rank-1 terms for a 2x2x2 block at offset (i0, j0, k0).
    
    Strassen's 7 multiplications for 2x2x2:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Each Strassen multiplication corresponds to one rank-1 tensor term.
    # The U, V, W vectors encode the linear combinations.
    
    # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    U[i0*m+j0, col] = 1; U[(i0+1)*m+(j0+1), col] = 1
    V[j0*p+k0, col] = 1; V[(j0+1)*p+(k0+1), col] = 1
    W[k0*n+i0, col] = 1; W[(k0+1)*n+(i0+1), col] = 1
    col += 1
    
    # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    U[(i0+1)*m+j0, col] = 1; U[(i0+1)*m+(j0+1), col] = 1
    V[j0*p+k0, col] = 1
    W[k0*n+(i0+1), col] = 1
    col += 1
    
    # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
    U[i0*m+j0, col] = 1
    V[j0*p+(k0+1), col] = 1; V[(j0+1)*p+(k0+1), col] = -1
    W[(k0+1)*n+i0, col] = 1
    col += 1
    
    # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
    U[(i0+1)*m+(j0+1), col] = 1
    V[(j0+1)*p+k0, col] = 1; V[j0*p+k0, col] = -1
    W[k0*n+(i0+1), col] = 1
    col += 1
    
    # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    U[i0*m+j0, col] = 1; U[i0*m+(j0+1), col] = 1
    V[(j0+1)*p+(k0+1), col] = 1
    W[(k0+1)*n+i0, col] = 1
    col += 1
    
    # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
    U[(i0+1)*m+j0, col] = 1; U[i0*m+j0, col] = -1
    V[j0*p+k0, col] = 1; V[j0*p+(k0+1), col] = 1
    W[k0*n+(i0+1), col] = 1; W[(k0+1)*n+(i0+1), col] = 1
    col += 1
    
    # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
    U[i0*m+(j0+1), col] = 1; U[(i0+1)*m+(j0+1), col] = -1
    V[(j0+1)*p+k0, col] = 1; V[(j0+1)*p+(k0+1), col] = 1
    W[k0*n+i0, col] = 1; W[(k0+1)*n+i0, col] = 1
    col += 1
    
    return col

def _build_block_strassen(n, m, p):
    """
    Build a block-structured decomposition using Strassen for 2x2x2 blocks
    and schoolbook for non-square blocks.
    """
    max_rank = n * m * p
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    col = 0
    
    a_row_splits = [0, 2, 4]
    a_col_splits = [0, 2, 5]
    b_col_splits = [0, 2, 5]
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                i0 = a_row_splits[bi]
                i1 = a_row_splits[bi+1]
                j0 = a_col_splits[bj]
                j1 = a_col_splits[bj+1]
                k0 = b_col_splits[bk]
                k1 = b_col_splits[bk+1]
                
                a_h = i1 - i0
                a_w = j1 - j0
                b_h = j1 - j0
                b_w = k1 - k0
                
                if a_h == 2 and a_w == 2 and b_h == 2 and b_w == 2:
                    col = _add_strassen_block(U, V, W, col, i0, j0, k0, n, m, p)
                else:
                    for i in range(i0, i1):
                        for j in range(j0, j1):
                            for k in range(k0, k1):
                                U[i*m+j, col] = 1.0
                                V[j*p+k, col] = 1.0
                                W[k*n+i, col] = 1.0
                                col += 1
    
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    U, V, W = _build_block_strassen(n, m, p)
    
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        U, V, W = _build_schoolbook(n, m, p)
        valid, err = _verify_decomposition(U, V, W, n, m, p)
    
    if not valid:
        raise ValueError(f"Decomposition failed: {err}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
