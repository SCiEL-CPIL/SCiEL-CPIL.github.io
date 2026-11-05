import numpy as np
import time

_START = time.time()
_TIME_LIMIT = 120.0

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[1, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_2x2x8():
    """
    Builds 2x2x8 decomposition.
    A is 2x2, B is 2x8, C is 2x8.
    Splits B into four 2x2 blocks.
    Uses four Strassen blocks.
    Rank 28.
    U shape: (4, 28), V shape: (16, 28), W shape: (16, 28)
    """
    U_str, V_str, W_str = _strassen_2x2x2()
    rank = 28
    U = np.zeros((4, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for block in range(4):
        offset = block * 7
        U[:, offset:offset+7] = U_str
        V[block*4:(block+1)*4, offset:offset+7] = V_str
        W[block*4:(block+1)*4, offset:offset+7] = W_str
    
    return U, V, W

def _build_2x4x8():
    """
    Builds 2x4x8 decomposition.
    A is 2x4, B is 4x8, C is 2x8.
    Splits A into A1, A2 (each 2x2).
    Splits B into B1, B2 (each 2x8).
    Uses two 2x2x8 blocks.
    Rank 56.
    U shape: (8, 56), V shape: (32, 56), W shape: (16, 56)
    """
    U_228, V_228, W_228 = _build_2x2x8()
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    # Block 1: A1 * B1
    U[0:4, 0:28] = U_228
    V[0:16, 0:28] = V_228
    W[0:16, 0:28] = W_228
    
    # Block 2: A2 * B2
    U[4:8, 28:56] = U_228
    V[16:32, 28:56] = V_228
    W[0:16, 28:56] = W_228
    
    return U, V, W

def _build_1x4x8():
    """
    Builds 1x4x8 decomposition.
    A is 1x4, B is 4x8, C is 1x8.
    Schoolbook: rank 32.
    U shape: (4, 32), V shape: (32, 32), W shape: (8, 32)
    """
    rank = 32
    U = np.zeros((4, rank))
    V = np.zeros((32, rank))
    W = np.zeros((8, rank))
    
    col = 0
    for j in range(4):
        for k in range(8):
            U[j, col] = 1
            V[j*8+k, col] = 1
            W[k, col] = 1
            col += 1
    
    return U, V, W

def _build_3x4x8():
    """
    Builds 3x4x8 decomposition.
    A is 3x4, B is 4x8, C is 3x8.
    Splits A into A_top (2x4) and A_bottom (1x4).
    Uses 2x4x8 (rank 56) and 1x4x8 (rank 32).
    Rank 88.
    U shape: (12, 88), V shape: (32, 88), W shape: (24, 88)
    """
    U_248, V_248, W_248 = _build_2x4x8()
    U_148, V_148, W_148 = _build_1x4x8()
    rank = 88
    U = np.zeros((12, rank))
    V = np.zeros((32, rank))
    W = np.zeros((24, rank))
    
    # Block 1: A_top * B (rows 0-1 of A)
    U[0:8, 0:56] = U_248
    V[0:32, 0:56] = V_248
    W[0:16, 0:56] = W_248
    
    # Block 2: A_bottom * B (row 2 of A)
    U[8:12, 56:88] = U_148
    V[0:32, 56:88] = V_148
    W[16:24, 56:88] = W_148
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_tests=10):
    """Verify that the decomposition correctly computes C = A @ B."""
    import random
    random.seed(42)
    rank = len(U[0]) if U else 0
    for _ in range(num_tests):
        A = np.random.randint(-5, 5, size=(n, m))
        B = np.random.randint(-5, 5, size=(m, p))
        
        true_C = A @ B
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        u = np.einsum('ij,imt->it', A, U_reshaped)
        v = np.einsum('jk,jpt->kt', B, V_reshaped)
        
        c = np.einsum('kit,it,kt->ik', W_reshaped, u, v)
        
        if not np.allclose(c, true_C, atol=1e-5):
            return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8."""
    n, m, p = 3, 4, 8
    
    # Deadline guard
    if time.time() - _START > _TIME_LIMIT - 5:
        return _schoolbook(n, m, p)
    
    # Try the rank 88 Strassen-based decomposition
    try:
        U, V, W = _build_3x4x8()
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass

    # Fallback to schoolbook (rank 96)
    return _schoolbook(n, m, p)
