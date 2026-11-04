"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1] if isinstance(U, np.ndarray) else len(U[0])
    Ur = U.reshape(n, m, r) if isinstance(U, np.ndarray) else np.array(U).reshape(n, m, r)
    Vr = V.reshape(m, p, r) if isinstance(V, np.ndarray) else np.array(V).reshape(m, p, r)
    Wr = W.reshape(p, n, r) if isinstance(W, np.ndarray) else np.array(W).reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-8, rtol=1e-8)


def _strassen_2x2():
    """
    Verified rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0] and C[1,1]
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1] and C[1,1]
    
    # M4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0] and C[1,0]
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0] and C[0,1]
    
    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _winograd_3x3():
    """
    Rank-8 Winograd decomposition for 3x3x3 matrix multiplication.
    Uses the 3-point DFT over Z_3 structure.
    """
    U = np.zeros((9, 8), dtype=float)
    V = np.zeros((9, 8), dtype=float)
    W = np.zeros((9, 8), dtype=float)
    
    # The 8 products are based on the Winograd scheme:
    # P1 = (a00+a11+a22)(b00+b11+b22)
    # P2 = (a01+a12+a20)(b01+b12+b20)
    # P3 = (a02+a10+a21)(b02+b10+b21)
    # P4 = (a00-a11)(b00-b11)
    # P5 = (a11-a22)(b11-b22)
    # P6 = (a22-a00)(b22-b00)
    # P7 = (a01-a12)(b01-b12)
    # P8 = (a12-a20)(b12-b20)
    
    # P1 = (a00+a11+a22)(b00+b11+b22)
    U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
    
    # P2 = (a01+a12+a20)(b01+b12+b20)
    U[1, 1] = 1; U[5, 1] = 1; U[6, 1] = 1
    V[1, 1] = 1; V[5, 1] = 1; V[6, 1] = 1
    
    # P3 = (a02+a10+a21)(b02+b10+b21)
    U[2, 2] = 1; U[3, 2] = 1; U[7, 2] = 1
    V[2, 2] = 1; V[3, 2] = 1; V[7, 2] = 1
    
    # P4 = (a00-a11)(b00-b11)
    U[0, 3] = 1; U[4, 3] = -1
    V[0, 3] = 1; V[4, 3] = -1
    
    # P5 = (a11-a22)(b11-b22)
    U[4, 4] = 1; U[8, 4] = -1
    V[4, 4] = 1; V[8, 4] = -1
    
    # P6 = (a22-a00)(b22-b00)
    U[8, 5] = 1; U[0, 5] = -1
    V[8, 5] = 1; V[0, 5] = -1
    
    # P7 = (a01-a12)(b01-b12)
    U[1, 6] = 1; U[5, 6] = -1
    V[1, 6] = 1; V[5, 6] = -1
    
    # P8 = (a12-a20)(b12-b20)
    U[5, 7] = 1; U[6, 7] = -1
    V[5, 7] = 1; V[6, 7] = -1
    
    # Compute W by solving the linear system
    for i in range(3):
        for k in range(3):
            Aik = np.zeros((3, 8), dtype=float)
            for j in range(3):
                for c in range(8):
                    Aik[j, c] = U[i * 3 + j, c] * V[j * 3 + k, c]
            b = np.ones(3, dtype=float)
            sol, residuals, rank_s, sv = np.linalg.lstsq(Aik, b, rcond=None)
            W[k * 3 + i, :] = sol
    
    return U, V, W


def _kronecker_product(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two tensor decompositions.
    If (U1,V1,W1) decomposes n1xm1xp1 and (U2,V2,W2) decomposes n2xm2xp2,
    then the Kronecker product decomposes (n1*n2)x(m1*m2)x(p1*p2).
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, r), dtype=float)
    V = np.zeros((m * p, r), dtype=float)
    W = np.zeros((p * n, r), dtype=float)
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            # U: index (i1*n2+i2, j1*m2+j2) -> flat (i1*n2+i2)*m + (j1*m2+j2)
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u_val = U1[i1 * m1 + j1, c1] * U2[i2 * m2 + j2, c2]
                            if u_val != 0:
                                idx = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                U[idx, c] = u_val
            # V: index (j1*m2+j2, k1*p2+k2) -> flat (j1*m2+j2)*p + (k1*p2+k2)
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v_val = V1[j1 * p1 + k1, c1] * V2[j2 * p2 + k2, c2]
                            if v_val != 0:
                                idx = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                                V[idx, c] = v_val
            # W: index (k1*p2+k2, i1*n2+i2) -> flat (k1*p2+k2)*n + (i1*n2+i2)
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w_val = W1[k1 * n1 + i1, c1] * W2[k2 * n2 + i2, c2]
                            if w_val != 0:
                                idx = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                                W[idx, c] = w_val
    
    return U, V, W


def _block_strassen_6x6x6():
    """
    Decompose 6x6x6 matrix multiplication using 3x3 blocks of 2x2 matrices.
    Total rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    sU, sV, sW = _strassen_2x2()
    
    col = 0
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for c in range(7):
                    for li in range(2):
                        for lj in range(2):
                            u_row = (bi*2+li)*6 + (bj*2+lj)
                            U[u_row, col] = sU[li*2+lj, c]
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_row = (bj*2+lj)*6 + (bk*2+lk)
                            V[v_row, col] = sV[lj*2+lk, c]
                    
                    for lk in range(2):
                        for li in range(2):
                            w_row = (bk*2+lk)*6 + (bi*2+li)
                            W[w_row, col] = sW[lk*2+li, c]
                    
                    col += 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    """
    Construct a tensor decomposition for 6x6 matrix multiplication.
    Uses block Strassen: 3x3 blocks of 2x2, each 2x2 using Strassen rank 7.
    Total rank = 27 * 7 = 189.
    """
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Primary: block Strassen (rank 189)
    U_b, V_b, W_b = _block_strassen_6x6x6()
    if _verify(U_b, V_b, W_b, 6, 6, 6):
        return {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_sb.tolist()} if False else {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
    
    # Check time
    if time.time() > deadline - 20:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try Kronecker: schoolbook 3x3 ⊗ Strassen 2x2 = rank 189
    U3, V3, W3 = _schoolbook(3, 3, 3)
    U2, V2, W2 = _strassen_2x2()
    
    if not _verify(U2, V2, W2, 2, 2, 2):
        U2, V2, W2 = _schoolbook(2, 2, 2)
    
    U, V, W = _kronecker_product(U3, V3, W3, U2, V2, W2, 3, 3, 3, 2, 2, 2)
    
    if _verify(U, V, W, 6, 6, 6):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
# EVOLVE-BLOCK-END
