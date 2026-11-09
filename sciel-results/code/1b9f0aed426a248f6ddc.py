import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 120.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 15.0)

def _strassen_2x2():
    """Strassen 2x2 decomposition (rank 7)."""
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1
    U2[0,2]=1; V2[3,2]=1; V2[1,2]=-1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1
    U2[3,6]=1; U2[2,6]=-1; V2[2,6]=1; V2[3,6]=1
    
    W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
    W2[1,2]=1; W2[1,4]=1
    W2[2,1]=1; W2[2,3]=1
    W2[3,0]=1; W2[3,1]=-1; W2[3,2]=1; W2[3,5]=1
    
    return U2, V2, W2

def _verify_decomposition(U, V, W, n, m, p, trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    for _ in range(trials):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            scalar = np.sum(u_vec * A) * np.sum(v_vec * B)
            if scalar != 0:
                w_mat = W[:, r].reshape(p, n).T
                C_computed += scalar * w_mat
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _schoolbook(n, m, p):
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

def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb):
    """Kronecker product of two tensor decompositions."""
    import math
    n1 = int(math.isqrt(len(Ua)))
    m1 = int(math.isqrt(len(Va)))
    p1 = int(math.isqrt(len(Wa)))
    n2 = int(math.isqrt(len(Ub)))
    m2 = int(math.isqrt(len(Vb)))
    p2 = int(math.isqrt(len(Wb)))
    r1 = Ua.shape[1]
    r2 = Ub.shape[1]
    n, m, p = n1*n2, m1*m2, p1*p2
    r = r1 * r2
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for r1i in range(r1):
        for r2i in range(r2):
            r_idx = r1i * r2 + r2i
            for i1 in range(n1):
                for j1 in range(m1):
                    a = Ua[i1*m1+j1, r1i]
                    if a == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            b = Ub[i2*m2+j2, r2i]
                            if b == 0: continue
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m+j, r_idx] = a * b
            for j1 in range(m1):
                for k1 in range(p1):
                    a = Va[j1*p1+k1, r1i]
                    if a == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            b = Vb[j2*p2+k2, r2i]
                            if b == 0: continue
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            V[j*p+k, r_idx] = a * b
            for k1 in range(p1):
                for i1 in range(n1):
                    a = Wa[k1*n1+i1, r1i]
                    if a == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            b = Wb[k2*n2+i2, r2i]
                            if b == 0: continue
                            k = k1*p2 + k2
                            i = i1*n2 + i2
                            W[k*n+i, r_idx] = a * b
    return U, V, W

def _laderman_3x3():
    """
    Laderman's 3x3 matrix multiplication (rank 23).
    Defines 23 bilinear forms (U,V) and solves for W via linear system.
    Returns None if verification fails.
    """
    U3 = np.zeros((9, 23))
    V3 = np.zeros((9, 23))
    W3 = np.zeros((9, 23))
    
    # 23 bilinear forms: each is (sum of a-indices with coeffs, sum of b-indices with coeffs)
    # Index: a[i,j] -> i*3+j, b[j,k] -> j*3+k
    products = [
        # P0:  (a00+a11)(b00+b11)
        ([0,4],[1,1], [0,4],[1,1]),
        # P1:  (a01+a11)(b00+b10)
        ([1,4],[1,1], [0,2],[1,1]),
        # P2:  (a10+a20)(b00+b01)
        ([3,6],[1,1], [0,1],[1,1]),
        # P3:  (a21+a11)(b10+b11)
        ([5,4],[1,1], [2,4],[1,1]),
        # P4:  (a00+a22)(b00+b22)
        ([0,8],[1,1], [0,8],[1,1]),
        # P5:  (a01+a21)(b02+b22)
        ([1,5],[1,1], [1,8],[1,1]),
        # P6:  (a02+a22)(b00+b20)
        ([2,8],[1,1], [0,6],[1,1]),
        # P7:  (a10+a22)(b02+b20)
        ([3,8],[1,1], [1,6],[1,1]),
        # P8:  (a12+a20)(b00+b22)
        ([4,6],[1,1], [0,8],[1,1]),
        # P9:  (a20+a02)(b11+b12)
        ([6,2],[1,1], [4,5],[1,1]),
        # P10: (a21+a01)(b10+b12)
        ([5,1],[1,1], [2,5],[1,1]),
        # P11: (a00+a12)(b01+b21)
        ([0,4],[1,1], [1,7],[1,1]),
        # P12: (a11+a22)(b00+b12)
        ([4,8],[1,1], [0,5],[1,1]),
        # P13: (a01+a22)(b00+b21)
        ([1,8],[1,1], [0,7],[1,1]),
        # P14: (a10+a02)(b11+b01)
        ([3,2],[1,1], [4,1],[1,1]),
        # P15: (a20+a12)(b10+b12)
        ([6,4],[1,1], [2,5],[1,1]),
        # P16: (a02+a10)(b11+b12)
        ([2,3],[1,1], [4,5],[1,1]),
        # P17: (a12+a21)(b00+b10)
        ([4,5],[1,1], [0,2],[1,1]),
        # P18: (a00+a22)(b01+b21)
        ([0,8],[1,1], [1,7],[1,1]),
        # P19: (a10+a21)(b00+b20)
        ([3,5],[1,1], [0,6],[1,1]),
        # P20: (a01+a10)(b11+b22)
        ([1,3],[1,1], [4,8],[1,1]),
        # P21: (a20+a22)(b10+b11)
        ([6,8],[1,1], [2,4],[1,1]),
        # P22: (a02+a12)(b21+b22)
        ([2,4],[1,1], [7,8],[1,1]),
    ]
    
    for r, (ui, uc, vi, vc) in enumerate(products):
        for idx, val in zip(ui, uc):
            U3[idx, r] = val
        for idx, val in zip(vi, vc):
            V3[idx, r] = val
    
    # Build the linear system for W.
    # For each output (i,k): sum_r W[k*3+i, r] * (sum_j U3[i*3+j,r]*V3[j*3+k,r]) = delta_{ik}
    # Let S[i,k,r] = sum_j U3[i*3+j,r]*V3[j*3+k,r]
    # Then for each (i,k): S[i,k,:] . W[k*3+i,:] = delta_{ik}
    # This is underdetermined (23 unknowns, 1 equation per (i,k)), so use minimum norm solution.
    
    S = np.zeros((9, 23))  # S[(i,k), r]
    for i in range(3):
        for k in range(3):
            for r in range(23):
                s = 0.0
                for j in range(3):
                    s += U3[i*3+j, r] * V3[j*3+k, r]
                S[i*3+k, r] = s
    
    # For each output position (i,k), solve: S[i*3+k, :] @ W[k*3+i, :] = delta_{ik}
    # Minimum norm solution: W = S_row^T / ||S_row||^2 * target
    for i in range(3):
        for k in range(3):
            row = S[i*3+k, :]
            target = 1.0 if i == k else 0.0
            norm_sq = np.dot(row, row)
            if norm_sq > 1e-12:
                W3[k*3+i, :] = row * (target / norm_sq)
    
    # Verify
    if not _verify_decomposition(U3, V3, W3, 3, 3, 3, trials=20):
        return None
    
    return U3, V3, W3

def _strassen_2x2():
    """Strassen 2x2 decomposition (rank 7)."""
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen's 7 products
    # M1=(a00+a11)(b00+b11)
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1
    # M2=(a10+a11)b00
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1
    # M3=a00(b11-b01)
    U2[0,2]=1; V2[3,2]=1; V2[1,2]=-1
    # M4=a11(b10-b00)
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1
    # M5=(a00+a01)b11
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1
    # M6=(a10-a00)(b00+b01)
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1
    # M7=(a11-a10)(b10+b11)
    U2[3,6]=1; U2[2,6]=-1; V2[2,6]=1; V2[3,6]=1
    
    # W: c00=M1+M4-M5+M7, c01=M3+M5, c10=M2+M4, c11=M1-M2+M3+M6
    W2[0,0]=1; W2[0,3]=1; W2[0,4]=-1; W2[0,6]=1
    W2[1,2]=1; W2[1,4]=1
    W2[2,1]=1; W2[2,3]=1
    W2[3,0]=1; W2[3,1]=-1; W2[3,2]=1; W2[3,5]=1
    
    return U2, V2, W2

def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb):
    """Kronecker product of two tensor decompositions."""
    import math
    n1 = int(math.isqrt(len(Ua)))
    m1 = int(math.isqrt(len(Va)))
    p1 = int(math.isqrt(len(Wa)))
    n2 = int(math.isqrt(len(Ub)))
    m2 = int(math.isqrt(len(Vb)))
    p2 = int(math.isqrt(len(Wb)))
    r1 = Ua.shape[1]
    r2 = Ub.shape[1]
    n, m, p = n1*n2, m1*m2, p1*p2
    r = r1 * r2
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for r1i in range(r1):
        for r2i in range(r2):
            r_idx = r1i * r2 + r2i
            # U: index (i1,i2,j1,j2) -> i = i1*n2+i2, j = j1*m2+j2
            for i1 in range(n1):
                for j1 in range(m1):
                    a = Ua[i1*m1+j1, r1i]
                    if a == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            b = Ub[i2*m2+j2, r2i]
                            if b == 0: continue
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m+j, r_idx] = a * b
            # V: index (j1,j2,k1,k2) -> j = j1*m2+j2, k = k1*p2+k2
            for j1 in range(m1):
                for k1 in range(p1):
                    a = Va[j1*p1+k1, r1i]
                    if a == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            b = Vb[j2*p2+k2, r2i]
                            if b == 0: continue
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            V[j*p+k, r_idx] = a * b
            # W: index (k1,k2,i1,i2) -> k = k1*p2+k2, i = i1*n2+i2
            for k1 in range(p1):
                for i1 in range(n1):
                    a = Wa[k1*n1+i1, r1i]
                    if a == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            b = Wb[k2*n2+i2, r2i]
                            if b == 0: continue
                            k = k1*p2 + k2
                            i = i1*n2 + i2
                            W[k*n+i, r_idx] = a * b
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    for _ in range(trials):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            scalar = np.sum(u_vec * A) * np.sum(v_vec * B)
            if scalar != 0:
                w_mat = W[:, r].reshape(p, n).T  # W[k*n+i] -> (i,k)
                C_computed += scalar * w_mat
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _schoolbook(n, m, p):
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

def _get_laderman_correct():
    """Get a verified correct Laderman 3x3 decomposition."""
    return _laderman_3x3()

def construct():
    n, m, p = 6, 6, 6
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 216 schoolbook
    
    # Strategy 1 (best): Laderman 3x3 (rank 23) ⊗ Strassen 2x2 (rank 7) = rank 161
    try:
        laderman = _laderman_3x3()
        if laderman is not None:
            U3, V3, W3 = laderman
            U2, V2, W2 = _strassen_2x2()
            U, V, W = _kron_decompose(U3, V3, W3, U2, V2, W2)
            if _verify_decomposition(U, V, W, 6, 6, 6, trials=5):
                best_U, best_V, best_W = U, V, W
                best_rank = 161
                print(f"Laderman3xStrassen2: rank {best_rank}")
    except Exception as e:
        print(f"Laderman+Strassen failed: {e}")
    
    # Strategy 2: Schoolbook 3x3 ⊗ Strassen 2x2 = rank 189
    if best_rank > 189:
        try:
            U3 = np.zeros((9, 27))
            V3 = np.zeros((9, 27))
            W3 = np.zeros((9, 27))
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U3[i*3+j, col] = 1
                        V3[j*3+k, col] = 1
                        W3[k*3+i, col] = 1
                        col += 1
            U2, V2, W2 = _strassen_2x2()
            U, V, W = _kron_decompose(U3, V3, W3, U2, V2, W2)
            if _verify_decomposition(U, V, W, 6, 6, 6, trials=3):
                best_U, best_V, best_W = U, V, W
                best_rank = 189
                print(f"Schoolbook3xStrassen2: rank {best_rank}")
        except Exception as e:
            print(f"Schoolbook3xStrassen2 failed: {e}")
    
    # Fallback: schoolbook
    if best_U is None:
        U, V, W = _schoolbook(6, 6, 6)
        best_U, best_V, best_W = U, V, W
        best_rank = 216
        print(f"Schoolbook fallback: rank {best_rank}")
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
