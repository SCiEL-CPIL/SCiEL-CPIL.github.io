import numpy as np
import time

DIMENSIONS = [4, 5, 5]
DEADLINE = 145.0  # Safety margin before 150s kill

def _strassen_2x2x2_embed(n, m, p, i0, j0, k0):
    """
    Returns U, V, W for a 2x2x2 block at offset (i0, j0, k0) in full n x m x p space.
    Uses Strassen's algorithm (rank 7).
    """
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    # M1 = (A00+A11)(B00+B11)
    U[ui(i0, j0), 0] = 1
    U[ui(i1, j1), 0] = 1
    V[vj(j0, k0), 0] = 1
    V[vj(j1, k1), 0] = 1
    W[wk(k0, i0), 0] = 1
    W[wk(k1, i1), 0] = 1
    
    # M2 = (A10+A11)B00
    U[ui(i1, j0), 1] = 1
    U[ui(i1, j1), 1] = 1
    V[vj(j0, k0), 1] = 1
    W[wk(k0, i1), 1] = 1
    W[wk(k1, i1), 1] = -1
    
    # M3 = A00(B01-B11)
    U[ui(i0, j0), 2] = 1
    V[vj(j0, k1), 2] = 1
    V[vj(j1, k1), 2] = -1
    W[wk(k1, i0), 2] = 1
    W[wk(k1, i1), 2] = 1
    
    # M4 = A11(B10-B00)
    U[ui(i1, j1), 3] = 1
    V[vj(j1, k0), 3] = 1
    V[vj(j0, k0), 3] = -1
    W[wk(k0, i0), 3] = 1
    W[wk(k0, i1), 3] = 1
    
    # M5 = (A00+A01)B11
    U[ui(i0, j0), 4] = 1
    U[ui(i0, j1), 4] = 1
    V[vj(j1, k1), 4] = 1
    W[wk(k0, i0), 4] = -1
    W[wk(k1, i0), 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[ui(i1, j0), 5] = 1
    U[ui(i0, j0), 5] = -1
    V[vj(j0, k0), 5] = 1
    V[vj(j0, k1), 5] = 1
    W[wk(k1, i1), 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[ui(i0, j1), 6] = 1
    U[ui(i1, j1), 6] = -1
    V[vj(j1, k0), 6] = 1
    V[vj(j1, k1), 6] = 1
    W[wk(k0, i0), 6] = 1
    
    return U, V, W

def _schoolbook_embed(n, m, p, i0, j0, k0, bi, bj, bk):
    """
    Returns U, V, W for a schoolbook bi x bj x bk block at offset (i0, j0, k0).
    """
    rank = bi * bj * bk
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(bi):
        for j in range(bj):
            for k in range(bk):
                U[(i0 + i) * m + (j0 + j), col] = 1
                V[(j0 + j) * p + (k0 + k), col] = 1
                W[(k0 + k) * n + (i0 + i), col] = 1
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using vectorized NumPy."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    # Build the full tensor contraction: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Vectorize over c
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Contract over rank dimension: T[i,j,k] = sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # T[i,j,k] should equal 1 for all i,j,k
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Decompose 4x5x5:
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # Split j into {0,1,2,3} and {4}:
    # Term 1a: A1(4x4) * B1a(4x4) -> C(4x4): i:0-3, j:0-3, k:0-3
    # Term 1b: A1(4x4) * B1b(4x1) -> C(4x1): i:0-3, j:0-3, k:4
    # Term 2:  A2(4x1) * B2(1x5)  -> C(4x5): i:0-3, j:4, k:0-4
    #
    # Term 1a: 8 Strassen 2x2x2 blocks -> rank 56
    # Term 1b: schoolbook -> rank 16
    # Term 2:  schoolbook -> rank 20
    # Total: 92
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # 8 Strassen 2x2x2 blocks for the 4x4x4 subproblem (i:0-3, j:0-3, k:0-3)
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                U_s, V_s, W_s = _strassen_2x2x2_embed(n, m, p, i0, j0, k0)
                U_parts.append(U_s)
                V_parts.append(V_s)
                W_parts.append(W_s)
    
    # Schoolbook for 4x4x1 (i:0-3, j:0-3, k:4)
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 0, 4, 4, 4, 1)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    # Schoolbook for 4x1x5 (i:0-3, j:4, k:0-4)
    U_s, V_s, W_s = _schoolbook_embed(n, m, p, 0, 4, 0, 4, 1, 5)
    U_parts.append(U_s)
    V_parts.append(V_s)
    W_parts.append(W_s)
    
    U_combined = np.hstack(U_parts)
    V_combined = np.hstack(V_parts)
    W_combined = np.hstack(W_parts)
    
    rank = U_combined.shape[1]
    
    # Verify using vectorized check
    if _verify(U_combined, V_combined, W_combined, n, m, p):
        return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # Fallback: pure schoolbook
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
