"""Matrix multiplication 4x4x8 tensor decomposition.
Constructs a rank-98 exact decomposition using the Kronecker product of
two Strassen 2x2x2 decompositions. Falls back to schoolbook rank-128
if verification fails.
"""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin


def _time_left():
    return max(0.0, _DEADLINE - time.time())


def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7.
    
    Conventions:
      U[i, r] : A factor, A indexed by (i_row, i_col) -> U[i_row*2 + i_col, r]
      V[j, r] : B factor, B indexed by (j_row, j_col) -> V[j_row*2 + j_col, r]
      W[k, r] : C factor, C indexed by (k_row, k_col) -> W[k_row*2 + k_col, r]
    """
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    # M3 = A00(B01-B11)
    U[0, 2] = 1; V[1, 2] = 1; V[3, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    # M4 = A11(B10-B00)
    U[3, 3] = 1; V[0, 3] = -1; V[2, 3] = 1; W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1; V[3, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1; V[2, 6] = 1; V[3, 6] = 1; W[0, 6] = 1
    return U, V, W


def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen.
    
    Returns U, V, W with shapes (16, 49), (16, 49), (16, 49).
    U indexed by (i, j) -> i*4 + j
    V indexed by (j, k) -> j*4 + k
    W indexed by (k, i) -> k*4 + i
    """
    U2, V2, W2 = _strassen_2x2x2()
    # Reshape to (2,2,7) for Kronecker-style construction
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    
    # For each pair of Strassen terms (c0, c1), the Kronecker product
    # U4[i,j,c] = U2[i//2, j//2, c0] * U2[i%2, j%2, c1]
    # This is equivalent to: for fixed c0,c1, U4[:, :, c] = U2[:, :, c0] ⊗ U2[:, :, c1]
    # where the Kronecker is over the (row, col) indices.
    
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            # Kronecker product of 2x2 matrices, reshaped to 4x4 then flattened row-major
            U4_block = np.kron(U2_3d[:, :, c0], U2_3d[:, :, c1])  # shape (4,4)
            V4_block = np.kron(V2_3d[:, :, c0], V2_3d[:, :, c1])  # shape (4,4)
            W4_block = np.kron(W2_3d[:, :, c0], W2_3d[:, :, c1])  # shape (4,4)
            U4[:, c] = U4_block.reshape(16)
            V4[:, c] = V4_block.reshape(16)
            W4[:, c] = W4_block.reshape(16)
    return U4, V4, W4


def _build_4x4x8_strassen():
    """Build 4x4x8 via two independent 4x4x4 Strassen-Kronecker blocks. Rank 98.
    
    U: (16, 98)
    V: (32, 98)
    W: (32, 98)
    """
    U4, V4, W4 = _build_4x4x4_tensors()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: columns 0-48
    U[:, 0:49] = U4
    # V for block 0: V[j*8 + k, c] = V4[j*4 + k, c] for k in 0..3
    V4_reshaped = V4.reshape(4, 4, 49)  # (j, k_local, c)
    V[0:32:8, 0:49] = V4_reshaped[:, 0, :]  # j=0, k=0
    V[1:32:8, 0:49] = V4_reshaped[:, 1, :]  # j=1, k=0
    # More efficiently: V[j*8 + k, c] for block 0 means V rows are j*8+k where k in [0,4)
    # Let's just do it directly
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
    
    # Block 1: columns 49-97
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:98] = W4[k * 4 + i, 0:49]
    return U, V, W


def _verify(U, V, W, n=4, m=4, p=8):
    """Verify the tensor decomposition exactly using vectorized operations.
    
    Checks if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
    """
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)  # (i, j, r)
    V_3d = V.reshape(m, p, rank)  # (j, k, r)
    W_3d = W.reshape(p, n, rank)  # (k, i, r)
    
    # For each (i,j,k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can compute this as: for each (i,j), compute U[i,j,:] * V[j,:,:,:] -> (p, r)
    # then element-wise with W and sum over r.
    # More efficiently: use einsum
    # result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    return np.all(result == 1)


def _schoolbook(n=4, m=4, p=8):
    """Fallback schoolbook decomposition, rank 128."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int16)
    V = np.zeros((m * p, rank), dtype=np.int16)
    W = np.zeros((p * n, rank), dtype=np.int16)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8."""
    # Try the rank-98 Strassen-Kronecker decomposition
    try:
        if _time_left() > 5:
            U, V, W = _build_4x4x8_strassen()
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback to schoolbook if Strassen fails or times out
    U, V, W = _schoolbook(4, 4, 8)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
