import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _build_schoolbook(n, m, p):
    """Build schoolbook rank decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Validate that U, V, W form an exact decomposition."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
    
    return np.allclose(T_approx, T, atol=tol)

def _validate_exact(U, V, W, n, m, p):
    """Exact validation using rational arithmetic."""
    # Convert to fractions
    U_f = [[Fraction(U[i,j]).limit_denominator(10000) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_f = [[Fraction(V[i,j]).limit_denominator(10000) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_f = [[Fraction(W[i,j]).limit_denominator(10000) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    rank = U.shape[1]
    T_approx = np.zeros((n*m, m*p, p*n), dtype=object)
    for a in range(n*m):
        for b in range(m*p):
            for d in range(p*n):
                s = Fraction(0)
                for c in range(rank):
                    s += U_f[a][c] * V_f[b][c] * W_f[d][c]
                T_approx[a,b,d] = s
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T_approx[i*m+j, j*p+k, k*n+i] != 1:
                    return False
                # Check that other entries are 0 (spot check)
    
    # More thorough check
    for a in range(n*m):
        for b in range(m*p):
            for d in range(p*n):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(d, n)
                if j == j2 and i == i2 and k == k2:
                    if T_approx[a,b,d] != 1:
                        return False
                else:
                    if T_approx[a,b,d] != 0:
                        return False
    return True

def _build_strassen_3x3():
    """Build a rank-19 or lower decomposition for 3x3 matrix multiplication.
    
    We'll use a known rank-19 algorithm if available, otherwise try to construct one.
    For now, let's use the schoolbook rank 27 for 3x3x3 and see if we can improve.
    
    Actually, let's try to use a block approach for 3x3 * 3x8.
    Split the 3x8 into three 3x3 blocks and one 3x2 block.
    """
    pass

def _try_block_decomposition(n, m, p):
    """Try to decompose 3x3 * 3x8 using block structure.
    
    Split B (3x8) into B1 (3x3), B2 (3x3), B3 (3x3), B4 (3x2).
    C = A * B = A*B1 + A*B2 + A*B3 + A*B4
    
    If we have rank-r1 decomposition for 3x3*3x3 and rank-r2 for 3x3*3x2,
    total rank = r1 + r2 (if we can share or if they're independent).
    
    Actually, the tensor is M_{3,3,8}. We can think of it as computing
    C[i,k] = sum_j A[i,j] B[j,k] for i=0..2, k=0..7.
    
    Let's try to find a direct decomposition.
    """
    pass

def _build_rank_decomposition_direct(n, m, p, r):
    """Try to build a rank-r decomposition directly."""
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find better decompositions
    # For 3x3 * 3x8, the rank is at least max(9, 24, 24) = 24
    # But practical lower bounds suggest higher.
    
    # Let's try a structured approach based on the structure of the problem.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # One approach: use the fact that we can group terms.
    # For each output (i,k), we need sum over j.
    # We can try to use shared products.
    
    # Let's try to find a decomposition with rank < 72.
    # We'll use a greedy approach or a known algorithm.
    
    # For now, let's just return the schoolbook solution to ensure correctness.
    # We can try to improve in subsequent iterations.
    
    if not _validate(best_U, best_V, best_W, n, m, p):
        # Fallback
        best_U, best_V, best_W = _build_schoolbook(n, m, p)
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
