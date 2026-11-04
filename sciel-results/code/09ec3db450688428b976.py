"""Winograd-style structured decomposition for 2x3 * 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _validate(U, V, W):
    """Check that the decomposition exactly reproduces the matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    A = rng.rand(n, m)
    B = rng.rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        Mc = U[:n*m, c].reshape(n, m)
        Nc = V[:m*p, c].reshape(m, p)
        wc = W[:p*n, c].reshape(p, n).T
        inner = Mc @ Nc
        C_computed += wc * inner
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def _make_rank17():
    """Construct a rank-17 decomposition using Strassen for the 2x2*2x2 sub-block."""
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    # U index: i*m + j = i*3 + j for A[i,j]
    # V index: j*p + k = j*3 + k for B[j,k]
    # W index: k*n + i = k*2 + i for C[i,k]
    
    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (terms 0-6)
    # A00=A[0,0]=U[0], A01=A[0,1]=U[1], A10=A[1,0]=U[3], A11=A[1,1]=U[4]
    # B00=B[0,0]=V[0], B01=B[0,1]=V[1], B10=B[1,0]=V[3], B11=B[1,1]=V[4]
    # C00=C[0,0]=W[0], C01=C[0,1]=W[1], C10=C[1,0]=W[2], C11=C[1,1]=W[3]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)*B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1  # C10 += M2, C11 += M2
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1; W[3][2] = -1  # C01 += M3, C11 -= M3
    
    # M4 = A11*(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1  # C00 -= M4, C10 += M4
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[3][4] = 1  # C01 += M5, C11 += M5
    
    # M6 = (A10-A00)*(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[2][5] = 1  # C00 += M6, C10 += M6
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[0][6] = 1  # C01 += M7, C00 += M7
    
    # Part 2: A[0:2,0:2] * B[0:2,2] -> C[0:2,2] (terms 7-10, schoolbook)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # B02=B[0,2]=V[2], B12=B[1,2]=V[5]
    # C02=C[0,2]=W[4], C12=C[1,2]=W[5]
    
    # Term 7: A00*B02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1  # C02
    
    # Term 8: A01*B12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1  # C02
    
    # Term 9: A10*B02
    U[3][9] = 1
    V[2][9] = 1
    W[5][9] = 1  # C12
    
    # Term 10: A11*B12
    U[4][10] = 1
    V[5][10] = 1
    W[5][10] = 1  # C12
    
    # Part 3: A[0:2,2] * B[2,0:3] -> C[0:2,0:3] (terms 11-16, outer product)
    # A02=A[0,2]=U[2], A12=A[1,2]=U[5]
    # B20=B[2,0]=V[6], B21=B[2,1]=V[7], B22=B[2,2]=V[8]
    
    # Term 11: A02*B20 -> C00
    U[2][11] = 1
    V[6][11] = 1
    W[0][11] = 1
    
    # Term 12: A02*B21 -> C01
    U[2][12] = 1
    V[7][12] = 1
    W[1][12] = 1
    
    # Term 13: A02*B22 -> C02
    U[2][13] = 1
    V[8][13] = 1
    W[4][13] = 1
    
    # Term 14: A12*B20 -> C10
    U[5][14] = 1
    V[6][14] = 1
    W[2][14] = 1
    
    # Term 15: A12*B21 -> C11
    U[5][15] = 1
    V[7][15] = 1
    W[3][15] = 1
    
    # Term 16: A12*B22 -> C12
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Return the rank-17 Strassen-based decomposition
    result = _make_rank17()
    
    return result
