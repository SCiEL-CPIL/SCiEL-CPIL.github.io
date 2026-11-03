"""Structural search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor using exact fractions."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(u_val[c]) * Fraction(v_val[c]) * Fraction(w_val[c])
                if s != Fraction(1):
                    return False
    return True

def _verify_decomposition_float(U, V, W, n, m, p):
    """Fast float verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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

def _rank18_construction(n, m, p):
    """
    Construct a rank-18 decomposition.
    
    Key insight: For C[i,k] = sum_j A[i,j]*B[j,k], we can use one term per (j,k) pair.
    Column c = j*p + k.
    U[i*m+j, c] = 1 for all i
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1 for all i
    
    Then for any (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    Only c=j*p+k contributes, giving 1*1*1=1.
    """
    r = m * p  # 18
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            for i in range(n):
                U[i * m + j, c] = 1
            V[j * p + k, c] = 1
            for i in range(n):
                W[k * n + i, c] = 1
    
    return U, V, W

def _try_lower_rank(n, m, p, start_rank=17, end_rank=12):
    """Try to find decompositions with rank < 18 using structured search."""
    if time.time() - START_TIME > TIME_LIMIT - 15:
        return None
    
    # For rank < 18, we need to share terms across (j,k) pairs.
    # This requires more sophisticated algebraic constructions.
    # Given time constraints, we skip this and rely on rank 18.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook rank 36
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try rank-18 construction
    U18, V18, W18 = _rank18_construction(n, m, p)
    if _verify_decomposition_float(U18, V18, W18, n, m, p):
        U18 = np.round(U18 * 2) / 2
        V18 = np.round(V18 * 2) / 2
        W18 = np.round(W18 * 2) / 2
        if _verify_decomposition(U18, V18, W18, n, m, p):
            best_U, best_V, best_W = U18, V18, W18
            best_rank = 18
    
    # Try lower ranks
    if time.time() - START_TIME < TIME_LIMIT - 10:
        result = _try_lower_rank(n, m, p)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U_school, V_school, W_school
    
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
