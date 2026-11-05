"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen. Rank 49."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

def _build_4x4x8_tensors():
    """
    Build 4x4x8 decomposition as direct sum of two 4x4x4 Strassen blocks.
    Total rank = 98.
    U: (16, 98)
    V: (32, 98)
    W: (32, 98)
    """
    U4, V4, W4 = _build_4x4x4_tensors()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: k in [0, 4), columns 0..48
    U[:, 0:49] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = k_local
            V[j*8 + k_global, 0:49] = V4[j*4 + k_local, :]
    for k_local in range(4):
        k_global = k_local
        for i in range(4):
            W[k_global*4 + i, 0:49] = W4[k_local*4 + i, :]
    
    # Block 1: k in [4, 8), columns 49..97
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local
            V[j*8 + k_global, 49:98] = V4[j*4 + k_local, :]
    for k_local in range(4):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global*4 + i, 49:98] = W4[k_local*4 + i, :]
    
    return U, V, W

def _verify_decomposition(U, V, W, n=4, m=4, p=8):
    """Verify the decomposition with random test matrices."""
    rank = len(U[0]) if U else 0
    np.random.seed(42)
    A = np.random.randint(-3, 4, size=(n, m)).astype(np.float64)
    B = np.random.randint(-3, 4, size=(m, p)).astype(np.float64)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p), dtype=np.float64)
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape to (rank, n, m), (rank, m, p), (rank, p, n)
    U_arr = np.array(U, dtype=np.float64).T.reshape(rank, n, m)
    V_arr = np.array(V, dtype=np.float64).T.reshape(rank, m, p)
    W_arr = np.array(W, dtype=np.float64).T.reshape(rank, p, n)
    
    for r in range(rank):
        Ar = A * U_arr[r]
        Br = B * V_arr[r]
        Mr = Ar @ Br
        Wr = W_arr[r].T  # (n, p)
        C_computed += Mr * Wr
    
    max_err = np.max(np.abs(C_computed - C_expected))
    return max_err < 1e-10

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    U, V, W = _build_4x4x8_tensors()
    
    # Verify if time permits
    if time.time() < _DEADLINE - 10:
        if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
            # Fallback to schoolbook rank-128 if verification fails
            n, m, p = 4, 4, 8
            rank = n * m * p
            U_f = np.zeros((n*m, rank), dtype=np.int16)
            V_f = np.zeros((m*p, rank), dtype=np.int16)
            W_f = np.zeros((p*n, rank), dtype=np.int16)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i*m*p + j*p + k
                        U_f[i*m+j, r] = 1
                        V_f[j*p+k, r] = 1
                        W_f[k*n+i, r] = 1
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
