import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=6, m=6, p=6, atol=1e-9):
    """Verify tensor decomposition: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    C_calc = np.zeros((n, p))
    U_arr = np.array(U) if isinstance(U, list) else U
    V_arr = np.array(V) if isinstance(V, list) else V
    W_arr = np.array(W) if isinstance(W, list) else W
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :])
                C_calc[i, k] += np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
    return np.allclose(C_expected, C_calc, atol=atol)

def _strassen_2x2():
    """Return (U, V, W) for 2x2 Strassen, rank 7."""
    R = 7
    U = np.zeros((4, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U[i,j,r]: coefficient of A[i,j] in the r-th product
    # V[j,k,r]: coefficient of B[j,k] in the r-th product
    # W[k,i,r]: coefficient in C[i,k]
    
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10 += 1, C11 -= 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B10-B11) -> C01 += 1, C11 += 1
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B01-B10) -> C00 += 1, C10 += 1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B10) -> C11 += 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B01+B11) -> C00 += 1
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_3x3():
    """Return (U, V, W) for 3x3 schoolbook, rank 27."""
    R = 27
    U = np.zeros((9, R))
    V = np.zeros((9, R))
    W = np.zeros((9, R))
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j, c] = 1
                V[j * 3 + k, c] = 1
                W[k * 3 + i, c] = 1
                c += 1
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two tensor decompositions.
    Block 1: n1 x m1 x p1 with rank R1
    Block 2: n2 x m2 x p2 with rank R2
    Result: (n1*n2) x (m1*m2) x (p1*p2) with rank R1*R2
    
    Indexing: i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    """
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R_total = R1 * R2
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    
    U_tot = np.zeros((n * m, R_total))
    V_tot = np.zeros((m * p, R_total))
    W_tot = np.zeros((p * n, R_total))
    
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            # U_tot[i1*n2+i2, j1*m2+j2, r] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2]
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1 * m1 + j1, r1]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2_val = U2[i2 * m2 + j2, r2]
                            if u2_val == 0:
                                continue
                            row_u = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                            U_tot[row_u, r] += u1_val * u2_val
            
            # V_tot[j1*m2+j2, k1*p2+k2, r] = V1[j1*p1+k1, r1] * V2[j2*p2+k2, r2]
            for j1 in range(m1):
                for k1 in range(p1):
                    v1_val = V1[j1 * p1 + k1, r1]
                    if v1_val == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2_val = V2[j2 * p2 + k2, r2]
                            if v2_val == 0:
                                continue
                            row_v = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                            V_tot[row_v, r] += v1_val * v2_val
            
            # W_tot[k1*p2+k2, i1*n2+i2, r] = W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2]
            for k1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[k1 * n1 + i1, r1]
                    if w1_val == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[k2 * n2 + i2, r2]
                            if w2_val == 0:
                                continue
                            row_w = (k1 * p2 + k2) * n + (i1 * n2 + i2)
                            W_tot[row_w, r] += w1_val * w2_val
    
    return U_tot, V_tot, W_tot

def _schoolbook_6x6():
    """Return (U, V, W) for 6x6 schoolbook, rank 216."""
    R = 216
    U = np.zeros((36, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))
    c = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                U[i * 6 + j, c] = 1
                V[j * 6 + k, c] = 1
                W[k * 6 + i, c] = 1
                c += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook 6x6 (rank 216)
    U_fb, V_fb, W_fb = _schoolbook_6x6()
    fallback = {
        "U": U_fb.tolist(),
        "V": V_fb.tolist(),
        "W": W_fb.tolist()
    }
    
    # Strategy: Kronecker product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7) = 189
    # This partitions 6x6 into a 2x2 grid of 3x3 blocks
    try:
        if time.time() > deadline - 10:
            return fallback
        
        U3, V3, W3 = _schoolbook_3x3()
        U2, V2, W2 = _strassen_2x2()
        
        # Verify 2x2 Strassen first
        if not _verify(U2, V2, W2, n=2, m=2, p=2):
            return fallback
        
        # Verify 3x3 schoolbook
        if not _verify(U3, V3, W3, n=3, m=3, p=3):
            return fallback
        
        if time.time() > deadline - 5:
            return fallback
        
        # Kronecker: 2x2 blocks of 3x3 sub-blocks
        # Block structure: 2x2 grid of 3x3 blocks
        # Outer: 2x2 (Strassen), Inner: 3x3 (schoolbook)
        # Total: 6x6
        U_tot, V_tot, W_tot = _kron_decomp(
            U3, V3, W3,   # inner: 3x3 schoolbook
            U2, V2, W2,   # outer: 2x2 Strassen
            n1=3, m1=3, p1=3,  # inner dimensions
            n2=2, m2=2, p2=2   # outer dimensions
        )
        
        if time.time() > deadline - 5:
            return fallback
        
        # Verify the full 6x6 decomposition
        if not _verify(U_tot, V_tot, W_tot, n=6, m=6, p=6):
            return fallback
        
        return {
            "U": U_tot.tolist(),
            "V": V_tot.tolist(),
            "W": W_tot.tolist()
        }
    except Exception:
        return fallback

# EVOLVE-BLOCK-END
