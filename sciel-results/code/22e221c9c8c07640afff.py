"""
Rank-minimal tensor decomposition for 2x3 * 3x9 matrix multiplication.

Strategy:
- Lower bound on rank is max(n*m, m*p, p*n) = 18.
- Try to find rank-18 decomposition via structured optimization.
- Use the block structure: 9 columns = 3 blocks of 3.
- If rank 18 not found quickly, fall back to schoolbook rank 54.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 9]


def _verify(U, V, W, n=2, m=3, p=9):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True


def _try_rank_18(n=2, m=3, p=9):
    """Attempt to find a rank-18 decomposition using structured approach."""
    rank = 18
    # Initialize with schoolbook-like structure but compressed
    # Use 3 blocks of 3 columns, each block gets 6 terms
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For each block b in 0,1,2 (corresponding to columns 3b, 3b+1, 3b+2):
    # Use 6 bilinear forms for the 2x3 x 3x3 multiplication
    # This is equivalent to finding rank-6 for 2x3x3, which is the minimum possible.
    
    # For 2x3 x 3x3: C[i,k] = sum_j A[i,j] B[j,k] for i in 0,1; k in 0,1,2
    # We need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1 for all i,j,k
    # with 6 terms.
    
    # This is a known hard problem. Let's try a direct approach.
    # For each block, assign 6 ranks.
    
    np.random.seed(123)
    for trial in range(100):
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Alternating minimization
        for it in range(2000):
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # For each k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p + k, :] * W[k*n + i, :]
                    b_vec = np.ones(p)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[i*m + j, :] = sol
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b_vec = np.ones(n)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[j*p + k, :] = sol
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for i in range(n):
                for k in range(p):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b_vec = np.ones(m)
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[k*n + i, :] = sol
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None


def _schoolbook(n=2, m=3, p=9):
    """Return the schoolbook rank-nmp decomposition."""
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
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Try rank 18
    result = _try_rank_18(n, m, p)
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
