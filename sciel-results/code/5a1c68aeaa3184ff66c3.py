import numpy as np
import time

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U: (i, j) for A[i,j]
    V: (j, k) for B[j,k]
    W: (i, k) for C[i,k]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[1, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_2x2x4():
    """
    Builds 2x2x4 decomposition.
    A is 2x2, B is 2x4, C is 2x4.
    Splits B into B1, B2 (2x2).
    Uses two Strassen blocks.
    Rank 14.
    U shape: (4, 14), V shape: (8, 14), W shape: (8, 14)
    """
    U_str, V_str, W_str = _strassen_2x2x2()
    rank = 14
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    # Block 1: B1 (cols 0-1 of B)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # In 2x4, B1 corresponds to rows 0,1 and cols 0,1.
    # V indices for B1: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U[:, 0:7] = U_str
    V[0:4, 0:7] = V_str
    W[0:4, 0:7] = W_str
    
    # Block 2: B2 (cols 2-3 of B)
    # B2 corresponds to rows 0,1 and cols 2,3.
    # V indices for B2: 4:(0,2), 5:(0,3), 6:(1,2), 7:(1,3)
    U[:, 7:14] = U_str
    V[4:8, 7:14] = V_str
    # W indices for C2: 4:(0,2), 5:(0,3), 6:(1,2), 7:(1,3)
    W[4:8, 7:14] = W_str
    
    return U, V, W

def _build_2x4x4():
    """
    Builds 2x4x4 decomposition.
    A is 2x4, B is 4x4, C is 2x4.
    Splits A into AL, AR (2x2).
    Splits B into BL, BR (2x4).
    Uses two 2x2x4 blocks.
    Rank 28.
    U shape: (8, 28), V shape: (16, 28), W shape: (8, 28)
    """
    U_224, V_224, W_224 = _build_2x2x4()
    rank = 28
    U = np.zeros((8, rank))
    V = np.zeros((16, rank))
    W = np.zeros((8, rank))
    
    # Block 1: AL * BL
    # AL is A[:, 0:2] -> U indices 0:4
    # BL is B[0:2, :] -> V indices 0:8
    # C1 is C[:, 0:2] -> W indices 0:4
    U[0:4, 0:14] = U_224
    V[0:8, 0:14] = V_224
    W[0:4, 0:14] = W_224
    
    # Block 2: AR * BR
    # AR is A[:, 2:4] -> U indices 4:8
    # BR is B[2:4, :] -> V indices 8:16
    # C2 is C[:, 2:4] -> W indices 4:8
    U[4:8, 14:28] = U_224
    V[8:16, 14:28] = V_224
    W[4:8, 14:28] = W_224
    
    return U, V, W

def _build_3x4x4():
    """
    Builds 3x4x4 decomposition.
    A is 3x4, B is 4x4, C is 3x4.
    Splits A into A2 (2x4) and A1 (1x4).
    Uses 2x4x4 (rank 28) and 1x4x4 (rank 16).
    Rank 44.
    U shape: (12, 44), V shape: (16, 44), W shape: (12, 44)
    """
    U_244, V_244, W_244 = _build_2x4x4()
    rank = 44
    U = np.zeros((12, rank))
    V = np.zeros((16, rank))
    W = np.zeros((12, rank))
    
    # Block 1: A2 * B (rows 0-1 of A)
    # U indices 0:8
    # V indices 0:16
    # W indices 0:8
    U[0:8, 0:28] = U_244
    V[0:16, 0:28] = V_244
    W[0:8, 0:28] = W_244
    
    # Block 2: A1 * B (row 2 of A)
    # Schoolbook for 1x4x4. Rank 16.
    # A1 is A[2, :] -> U indices 8:12 (j=0..3)
    # B is B[j, k] -> V indices j*4+k
    # C1 is C[2, :] -> W indices 8:12 (k=0..3)
    for j in range(4):
        for k in range(4):
            idx = 28 + j * 4 + k
            U[8 + j, idx] = 1
            V[j * 4 + k, idx] = 1
            W[8 + k, idx] = 1
            
    return U, V, W

def _build_3x4x8():
    """
    Builds 3x4x8 decomposition.
    A is 3x4, B is 4x8, C is 3x8.
    Splits B into B1, B2 (4x4).
    Uses two 3x4x4 blocks.
    Rank 88.
    U shape: (12, 88), V shape: (32, 88), W shape: (24, 88)
    """
    U_344, V_344, W_344 = _build_3x4x4()
    rank = 88
    U = np.zeros((12, rank))
    V = np.zeros((32, rank))
    W = np.zeros((24, rank))
    
    # Block 1: A * B1 (cols 0-3 of B)
    # U indices 0:12
    # V indices 0:16 (B1 is 4x4)
    # W indices 0:12 (C1 is 3x4)
    U[:, 0:44] = U_344
    V[0:16, 0:44] = V_344
    W[0:12, 0:44] = W_344
    
    # Block 2: A * B2 (cols 4-7 of B)
    # U indices 0:12
    # V indices 16:32 (B2 is 4x4)
    # W indices 12:24 (C2 is 3x4)
    U[:, 44:88] = U_344
    V[16:32, 44:88] = V_344
    W[12:24, 44:88] = W_344
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    """Verify that the decomposition correctly computes C = A @ B."""
    import random
    random.seed(42)
    rank = len(U[0]) if U else 0
    for _ in range(num_tests):
        A = np.random.randint(-5, 5, size=(n, m))
        B = np.random.randint(-5, 5, size=(m, p))
        
        true_C = A @ B
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        u = np.einsum('ij,imt->it', A, U_reshaped)
        v = np.einsum('jk,jpt->kt', B, V_reshaped)
        
        c = np.einsum('kit,it,kt->ik', W_reshaped, u, v)
        
        if not np.allclose(c, true_C, atol=1e-5):
            return False
    return True

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8."""
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Try to build the rank 88 decomposition
    try:
        if time.time() < deadline:
            U, V, W = _build_3x4x8()
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass

    # Fallback to schoolbook (rank 96)
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
