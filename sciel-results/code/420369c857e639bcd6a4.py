import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 130

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2 matrix multiplication. Rank 7."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _conv_1x2x3():
    """
    Schoolbook rank-4 decomposition for 1x2x3 linear convolution.
    n=1, m=2, p=3.
    U: (2, 4), V: (6, 4), W: (3, 4)
    Trace convention: W[k*n+i, r] = W[k, r] since n=1.
    """
    U = np.zeros((2, 4), dtype=np.float64)  # (n*m, rank) = (2, 4)
    V = np.zeros((6, 4), dtype=np.float64)  # (m*p, rank) = (6, 4)
    W = np.zeros((3, 4), dtype=np.float64)  # (p*n, rank) = (3, 4)
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U[0, 0] = 1  # i=0, j=0
    V[0, 0] = 1  # j=0, k=0
    W[0, 0] = 1  # k=0, i=0
    
    # Term 1: A[0,1]*B[1,0] -> C[0,1]
    U[1, 1] = 1  # i=0, j=1
    V[3, 1] = 1  # j=1, k=0 -> index j*p+k = 1*3+0 = 3
    W[1, 1] = 1  # k=1, i=0
    
    # Term 2: A[1,0]*B[0,1] -> C[0,1]
    U[0, 2] = 1  # i=0, j=0
    V[1, 2] = 1  # j=0, k=1 -> index 0*3+1 = 1
    W[1, 2] = 1  # k=1, i=0
    
    # Term 3: A[1,1]*B[1,1] -> C[0,2]
    U[1, 3] = 1  # i=0, j=1
    V[5, 3] = 1  # j=1, k=1 -> index 1*3+1 = 5
    W[2, 3] = 1  # k=2, i=0
    
    return U, V, W

def _build_wga_2x4x6():
    """
    Build 2x4x6 decomposition via WGA: (2x2x2) ⊗ (1x2x3).
    Rank = 7 * 4 = 28.
    
    Index mapping:
    i = i1 (i1 in {0,1}, i2=0)
    j = 2*j1 + j2 (j1 in {0,1}, j2 in {0,1})
    k = 3*k1 + k2 (k1 in {0,1}, k2 in {0,1,2})
    r = 4*r1 + r2 (r1 in {0..6}, r2 in {0..3})
    """
    n, m, p = 2, 4, 6
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    Us, Vs, Ws = _strassen_2x2x2()
    Uc, Vc, Wc = _conv_1x2x3()
    
    for r1 in range(7):
        for r2 in range(4):
            r = 4 * r1 + r2
            
            # U: i = i1, j = 2*j1 + j2
            for i1 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        i = i1
                        j = 2 * j1 + j2
                        U[i * m + j, r] = Us[i1 * 2 + j1, r1] * Uc[0 * 2 + j2, r2]
            
            # V: j = 2*j1 + j2, k = 3*k1 + k2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            j = 2 * j1 + j2
                            k = 3 * k1 + k2
                            V[j * p + k, r] = Vs[j1 * 2 + k1, r1] * Vc[j2 * 3 + k2, r2]
            
            # W: k = 3*k1 + k2, i = i1
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        k = 3 * k1 + k2
                        i = i1
                        W[k * n + i, r] = Ws[k1 * 2 + i1, r1] * Wc[k2 * 1 + 0, r2]
    
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    n, m, p = 2, 4, 6
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    error = np.linalg.norm(target - recon)
    return error < tol, error

def construct():
    # Build WGA rank-28 decomposition
    U, V, W = _build_wga_2x4x6()
    
    # Verify
    valid, error = _verify(U, V, W)
    
    if not valid:
        # Fallback: schoolbook rank 48
        rank = 48
        U = np.zeros((8, rank), dtype=np.float64)
        V = np.zeros((24, rank), dtype=np.float64)
        W = np.zeros((12, rank), dtype=np.float64)
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    U[i * 4 + j, col] = 1
                    V[j * 6 + k, col] = 1
                    W[k * 2 + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
