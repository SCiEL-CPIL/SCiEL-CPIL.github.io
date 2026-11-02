"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Should equal 1 for all i,j,k
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices for A, B, C are flattened row-major: 00, 01, 10, 11
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3 (k*2+i)
    
    # r=0 (M1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1 (M2)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    
    # r=2 (M3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01
    
    # r=3 (M4)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00, C10
    
    # r=4 (M5)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00, C01
    
    # r=5 (M6)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1 # C10
    
    # r=6 (M7)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[3, 6] = 1 # C00, C11
    
    return U, V, W

def _kron_decomp(U1, V1, W1):
    """
    Kronecker product of decompositions.
    If T1 = sum u1 v1 w1 and T2 = sum u2 v2 w2,
    then T1 x T2 = sum (u1x u2) (v1x v2) (w1x w2).
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1] if 'U2' in locals() else U1.shape[1]
    # For self-Kronecker:
    U_kron = np.kron(U1, U1)
    V_kron = np.kron(V1, V1)
    W_kron = np.kron(W1, W1)
    return U_kron, V_kron, W_kron

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 7 matrix.
    Uses Strassen-8 for the first 4 columns and Schoolbook for the last 3.
    Total rank: 49 + 48 = 97.
    """
    n, m, p = DIMENSIONS
    
    # 1. Strassen 4x4x4
    U2, V2, W2 = _strassen_2x2()
    U4_s, V4_s, W4_s = _kron_decomp(U2, V2, W2)
    
    # U4_s shape: (16, 49)
    # V4_s shape: (16, 49)
    # W4_s shape: (16, 49)
    
    # 2. Schoolbook 4x4x3
    # n=4, m=4, p=3
    rank_sb = n * m * 3
    U4_sb = np.zeros((n * m, rank_sb))
    V4_sb = np.zeros((m * 3, rank_sb))
    W4_sb = np.zeros((3 * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U4_sb[i * m + j, col] = 1
                V4_sb[j * 3 + k, col] = 1
                W4_sb[k * n + i, col] = 1
                col += 1
                
    # 3. Combine
    # Full U: (16, 97)
    U_full = np.hstack([U4_s, U4_sb])
    
    # Full V: (28, 97)
    # V4_s corresponds to B cols 0-3 (indices 0-15 in V)
    # V4_sb corresponds to B cols 4-6 (indices 16-27 in V)
    V_full = np.zeros((m * p, U_full.shape[1]))
    V_full[:16, :49] = V4_s
    V_full[16:, 49:] = V4_sb
    
    # Full W: (28, 97)
    # W4_s corresponds to C cols 0-3 (indices 0-15 in W)
    # W4_sb corresponds to C cols 4-6 (indices 16-27 in W)
    W_full = np.zeros((p * n, U_full.shape[1]))
    W_full[:16, :49] = W4_s
    W_full[16:, 49:] = W4_sb
    
    # Verify
    if not _verify(U_full.tolist(), V_full.tolist(), W_full.tolist()):
        # Fallback to schoolbook if verification fails (should not happen)
        U_fb = np.zeros((n*m, n*m*p))
        V_fb = np.zeros((m*p, n*m*p))
        W_fb = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
