import numpy as np
import time

DIMENSIONS = [4, 4, 9]
_DEADLINE = None

def _strassen_2x2():
    """Strassen 2x2x2, rank 7. U(4,7),V(4,7),W(4,7). W[j*2+i]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1=(A00+A11)(B00+B11) -> C00+C11
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2=(A10+A11)B00 -> C10-C11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3=A00(B01-B11) -> C01+C11
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4=A11(B10-B00) -> C10+C00
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5=(A00+A01)B11 -> C00-C10
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6=(A10-A00)(B00+B01) -> C11
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7=(A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _strassen_4x4x4():
    """Strassen 4x4x4 via 2-level, rank 49. U(16,49),V(16,49),W(16,49)."""
    U2, V2, W2 = _strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    big = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for bi, (A_b, B_b, C_b) in enumerate(big):
        for si in range(7):
            r = bi*7 + si
            for (bi_i, bi_k, ca) in A_b:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2*bi_i + i2; k = 2*bi_k + k2
                        U[i*4+k, r] += ca * U2[i2*2+k2, si]
            for (bk_i, bk_j, cb) in B_b:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2*bk_i + k2; j = 2*bk_j + j2
                        V[k*4+j, r] += cb * V2[k2*2+j2, si]
            for (cj_i, cj_j, cc) in C_b:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2*cj_i + j2; i = 2*cj_j + i2
                        W[j*4+i, r] += cc * W2[j2*2+i2, si]
    return U, V, W

def _build_4x4x9():
    """
    4x4 x 4x9 -> 4x9.
    Split B into B1(4x4), B2(4x4), B3(4x1).
    Use Strassen rank 49 for B1, B2; direct rank 16 for B3.
    Total rank = 49+49+16 = 114.
    """
    U4, V4, W4 = _strassen_4x4x4()
    rank = 49 + 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Block 1: cols 0-3
    U[:, 0:49] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:49] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:49] = W4[j*4+i, :]
    
    # Block 2: cols 4-7
    U[:, 49:98] = U4
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), 49:98] = V4[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, 49:98] = W4[j*4+i, :]
    
    # Block 3: col 8 (direct, rank 16)
    for i in range(4):
        for k in range(4):
            r = 98 + i*4 + k
            U[i*4+k, r] = 1
            V[k*9+8, r] = 1
            W[8*4+i, r] = 1
    
    return U, V, W, rank

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 140
    
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_4x4x9()
    
    # Verify with deterministic test matrices
    np.random.seed(42)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 9)
    C_exp = A @ B
    
    Ur = U.reshape(4, 4, rank)
    Vr = V.reshape(4, 9, rank)
    Wr = W.reshape(9, 4, rank)
    
    a = np.einsum('ikr,ik->ir', Ur, A)
    b = np.einsum('kjr,kj->jr', Vr, B)
    C = np.einsum('jir,ir,jr->ij', Wr, a, b)
    
    if not np.allclose(C, C_exp, atol=1e-10):
        # Fallback: direct rank 144
        rank = 144
        U = np.zeros((16, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        idx = 0
        for i in range(4):
            for j in range(9):
                for k in range(4):
                    U[i*4+k, idx] = 1
                    V[k*9+j, idx] = 1
                    W[j*4+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
